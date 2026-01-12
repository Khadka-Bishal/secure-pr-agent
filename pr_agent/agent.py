"""Workflow Orchestrator module.

Orchestrates the Pull Request review process using a deterministic state graph.
"""

import logging
import os
import textwrap
from typing import Any, Dict, List, Union

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from pr_agent.git import GitHubClient
from pr_agent.llm import LLMClient
from pr_agent.rag import ChromaRetriever, CodeChunker
from pr_agent.sandbox import SandboxRunner
from pr_agent.utils import get_tracer
from pr_agent.utils.mermaid_validation import validate_mermaid_structure

tracer = get_tracer()
logger = logging.getLogger("Workflow")


class CodeChunk(BaseModel):
    """Represents a discrete unit of code for analysis."""

    id: str
    content: str
    filepath: str


class AgentState(BaseModel):
    """Represents the shared state of the review workflow.

    Attributes:
        pr_url: Target Pull Request URL.
        diff_content: Raw diff content.
        chunks: List of code chunks extracted from the PR.
        relevant_chunks: Chunks deemed relevant for deep analysis.
        findings: List of security/bug findings.
        reflection_score: Self-assessed quality score (0-10).
        final_report: Markdown report to be posted.
        reviewed_commit_sha: The SHA of the commit being reviewed.
        status: Workflow status (running, skipped, completed).
    """

    pr_url: str
    diff_content: str = ""
    chunks: List[CodeChunk] = Field(default_factory=list)
    relevant_chunks: List[CodeChunk] = Field(default_factory=list)
    findings: List[str] = Field(default_factory=list)
    reflection_score: int = 0
    reflection_retries: int = 0
    final_report: str = ""
    reviewed_commit_sha: str = ""
    status: str = "running"
    # Mermaid Diagram State
    diagram_code: str = ""
    diagram_valid: bool = False
    diagram_retries: int = 0


class PullRequestAgent:
    """Autonomous agent that reviews PRs using a RAG-enhanced LLM workflow."""

    def __init__(
        self,
        llm: LLMClient,
        github: GitHubClient,
        collection_name: str = "main",
        force: bool = False,
    ):
        """Initialize the PullRequestAgent."""
        self.llm = llm
        self.github = github
        self.rag = CodeChunker()
        self.chroma = ChromaRetriever(
            model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            collection_name=collection_name,
        )
        self.sandbox = SandboxRunner()
        self.app = self._build_workflow()
        self.force_review = force

        logger.info(
            f"Initialized Agent with Read-Only Context from collection: '{collection_name}'\n"
        )

    def run(self, pr_url: str) -> Dict[str, Any]:
        """Executes the complete review workflow for a given PR."""
        from pr_agent.utils.logger import REVIEW_DURATION

        initial_state = AgentState(pr_url=pr_url)

        with tracer.start_as_current_span("workflow_run") as span:
            span.set_attribute("pr_url", pr_url)
            runner_type = os.getenv("SANDBOX_TYPE", "docker")
            repo_name = pr_url.rstrip("/").split("/")[-1]
            with REVIEW_DURATION.labels(repo=repo_name, runner_type=runner_type).time():
                return self.app.invoke(initial_state)

    def _build_workflow(self) -> StateGraph:
        """Constructs the state graph for the agent."""
        workflow = StateGraph(AgentState)

        workflow.add_node("fetch_context", self.fetch_context)
        workflow.add_node("analyze_code", self.analyze_code)
        workflow.add_node("self_reflect", self.self_reflect)
        workflow.add_node("generate_tests", self.generate_tests)

        # Diagram Nodes
        workflow.add_node("generate_diagram", self.generate_diagram)
        workflow.add_node("validate_diagram", self.validate_diagram)

        workflow.add_node("post_to_github", self.post_to_github)

        workflow.set_entry_point("fetch_context")

        workflow.add_conditional_edges(
            "fetch_context",
            self.check_context,
            {"continue": "analyze_code", "skip": END},
        )

        workflow.add_edge("analyze_code", "self_reflect")

        workflow.add_conditional_edges(
            "self_reflect",
            self.check_reflection,
            {"retry": "increment_reflection_retry", "good": "generate_tests"},
        )

        # Helper node to persist retry count
        workflow.add_node(
            "increment_reflection_retry", self.increment_reflection_retry
        )
        workflow.add_edge("increment_reflection_retry", "analyze_code")

        # After tests, try to generate diagram before posting
        workflow.add_edge("generate_tests", "generate_diagram")
        workflow.add_edge("generate_diagram", "validate_diagram")

        workflow.add_conditional_edges(
            "validate_diagram",
            self.check_diagram_validity,
            {
                "retry": "validate_diagram",
                "valid": "post_to_github",
                "give_up": "post_to_github",
            },
        )

        workflow.add_edge("post_to_github", END)

        return workflow.compile()

    def fetch_context(self, state: AgentState) -> Dict[str, Any]:
        """Fetches PR metadata and extracts code chunks."""
        logger.info("Fetching Context...")
        pr, files = self.github.get_pr_diff(state.pr_url, force=self.force_review)

        if files is None:
            logger.info("Idempotency hit: Review already exists. Skipping.")
            return {"status": "skipped", "final_report": "SKIPPED_IDEMPOTENCY"}

        if not files:
            logger.info("No file changes detected in PR diff.")
            return {"diff_content": ""}

        all_chunks = []
        commit_sha = ""
        for file in files:
            if not commit_sha and "sha" in file:
                commit_sha = file["sha"]
            chunks_data = self.rag.chunk_code(file["filename"], file["patch"])
            for c in chunks_data:
                all_chunks.append(
                    CodeChunk(
                        id=c["id"], content=c["content"], filepath=file["filename"]
                    )
                )

        return {
            "chunks": all_chunks,
            "diff_content": str(files),
            "reviewed_commit_sha": commit_sha,
        }

    def check_context(self, state: AgentState) -> str:
        """Determines if the workflow should proceed or skip."""
        if state.status == "skipped":
            return "skip"
        return "continue"

    def analyze_code(self, state: AgentState) -> Dict[str, Any]:
        """Analyzes code chunks using RAG-enriched LLM prompts."""
        logger.info("Analyzing Code...")

        findings = []
        for chunk in state.chunks:
            logger.debug(f"Processing chunk: {chunk.id}")

            external_context = self.chroma.retrieve_external_context(chunk.content)

            # RAG Transparency Logging
            if external_context:
                logger.info(
                    f"  [RAG] Retrieved {len(external_context)} context items for "
                    f"'{chunk.id}':"
                )
                for node in external_context:
                    logger.info(
                        f"    - Found dependency in "
                        f"{node.get('filepath', 'unknown')} ({node['id']})"
                    )
            else:
                logger.info(f"  [RAG] No relevant context found for '{chunk.id}'")

            context_str = self._format_context(external_context)

            full_payload = f"{chunk.content}{context_str}"
            finding = self.llm.review_code_chunk(full_payload, "python")
            findings.append(finding)

        return {"findings": findings, "relevant_chunks": state.chunks}

    def _format_context(self, context_nodes: List[Dict]) -> str:
        """Formats retrieved context for the LLM prompt."""
        if not context_nodes:
            return ""

        formatted = "\n\n### External Context:\n"
        for node in context_nodes:
            summary = node.get("content", "").splitlines()[0]
            formatted += f"- {node['id']} ({summary})\n"
        return formatted

    def self_reflect(self, state: AgentState) -> Dict[str, int]:
        """Evaluates the quality of the generated findings."""
        logger.info("Self-Reflection...")
        if not state.findings:
            return {"reflection_score": 10}

        latest_findings = state.findings[-1].strip()

        if latest_findings == "PASS":
            logger.info("Audit Result: PASS (Deterministic Quality: 10/10)")
            return {"reflection_score": 10}

        prompt = textwrap.dedent(f"""
            Act as a Quality Auditor.
            Review Content: "{latest_findings}"
            
            Rate the quality of this security review (1-10).
            High Score (8-10): Only critical bugs mentioned, extremely concise, no fluff.
            Low Score (1-4): Verbose, mentions nitpicks.
            
            Return ONLY the integer.
        """)

        rating = self.llm.chat(
            [
                {
                    "role": "system",
                    "content": "You are a logic gate. Return exactly one integer.",
                },
                {"role": "user", "content": prompt},
            ]
        )

        try:
            score = int(rating.strip())
        except ValueError:
            logger.warning(
                f"Failed to parse reflection score from '{rating}', defaulting to 5."
            )
            score = 5

        return {"reflection_score": score}

    def check_reflection(self, state: AgentState) -> str:
        """Determines workflow branching based on quality score."""
        score = state.reflection_score
        if score < 7:
            if state.reflection_retries >= 2:
                logger.warning(
                    f"Quality Score {score}/10 is insufficient, but Max Retries (2) reached. Proceeding anyway."
                )
                return "good"

            logger.info(
                f"Quality Score {score}/10 is insufficient. Retrying analysis..."
            )
            return "retry"

        logger.info(f"Quality Score {score}/10 is acceptable.")
        return "good"

    def increment_reflection_retry(self, state: AgentState) -> Dict[str, Any]:
        """Node to persist the retry count in state."""
        return {"reflection_retries": state.reflection_retries + 1}

    def generate_tests(self, state: AgentState) -> Dict[str, Any]:
        """Generates and executes verification scripts in the sandbox."""
        logger.info("Dynamic Verification...")

        if not state.findings:
            return {"final_report": "No findings to verify."}

        findings_str = "\n".join(state.findings)
        prompt = textwrap.dedent(f"""
            Act as a Senior QA Automation Engineer.
            Goal: Verify the following code review findings using a self-contained Python script.
            
            Findings:
            {findings_str}
            
            Instructions:
            1. For BUGS/SECURITY: Write a reproduction script to trigger the issue.
               - Print "Verification: FAILURE TRIGGERED" on success.
            2. For CLEAN CODE: Write a smoke test (import + basic execution).
               - Print "Verification: SUCCESS" on success.
            
            Constraints:
            - Return ONLY raw Python code (no markdown).
            - Must be purely self-contained (mock standard libs if needed).
        """)

        raw_response = self.llm.chat(
            [
                {
                    "role": "system",
                    "content": "You are a QA Engineer. Return only python code.",
                },
                {"role": "user", "content": prompt},
            ]
        )

        test_code = raw_response.replace("```python", "").replace("```", "").strip()

        logger.info(f"\n--- [Generated Verification Script] ---\n{test_code}\n---------------------------------------\n")

        runner_type = os.getenv("SANDBOX_TYPE", "docker")
        result = self.sandbox.run_tests(test_code, runner_type=runner_type)

        if "Sandbox Execution Failed" in result:
            logger.error(f"Verification Failed: {result}")
        else:
            logger.info(f"Verification Result:\n{result}")

        return {}

    def generate_diagram(self, state: AgentState) -> Dict[str, Any]:
        """Generates a Mermaid diagram from the diff content."""
        logger.info("Generating Diagram...")

        if not state.chunks:
            return {"diagram_valid": True}

        code = self.llm.generate_diagram(state.diff_content)
        clean_code = code.replace("```mermaid", "").replace("```", "").strip()

        return {"diagram_code": clean_code, "diagram_retries": 0}

    def validate_diagram(self, state: AgentState) -> Dict[str, Any]:
        """Validates the generated diagram syntax using a Deterministic Linter."""
        logger.info(f"Validating Diagram (Attempt {state.diagram_retries + 1})")

        if not state.diagram_code:
            return {"diagram_valid": True}

        is_valid, message = validate_mermaid_structure(state.diagram_code)

        if is_valid:
            logger.info("Diagram Syntax: VALID.")
            return {"diagram_valid": True}
        else:
            logger.info(f"Diagram Syntax: INVALID ({message}) -> Requesting Fix...")

            fixed_code = self.llm.fix_diagram_syntax(state.diagram_code, message)
            clean_code = (
                fixed_code.replace("```mermaid", "").replace("```", "").strip()
            )

            return {
                "diagram_code": clean_code,
                "diagram_valid": False,
                "diagram_retries": state.diagram_retries + 1,
            }

    def check_diagram_validity(self, state: AgentState) -> str:
        """Determines if we should retry diagram generation."""
        if state.diagram_valid:
            return "valid"

        if state.diagram_retries >= 2:
            logger.warning(
                "Diagram validation failed after 2 retries. Skipping diagram."
            )
            return "give_up"

        return "retry"

    def post_to_github(self, state: AgentState) -> Dict[str, str]:
        """Publishes the final report to the PR."""
        from pr_agent.utils.logger import FINDINGS_COUNT

        logger.info("Publishing Report...")

        # Filter out "PASS" or empty findings
        valid_findings = [f for f in state.findings if f and "PASS" not in f]

        report = "## PR AGENT REVIEW\n"

        if not valid_findings:
            report += "\n✅ **No critical issues found.**\n\nCode analysis passed all security checks.\n"
        else:
            for finding in valid_findings:
                report += f"\n{finding}\n"
                FINDINGS_COUNT.labels(severity="critical", category="security").inc()

        if state.diagram_code and state.diagram_valid:
            report += "\n### Visual Summary\n"
            report += f"```mermaid\n{state.diagram_code}\n```\n"

        if state.reviewed_commit_sha:
            report += f"\n\n*(Reviewed Commit: {state.reviewed_commit_sha})*"

        self.github.post_comment(state.pr_url, report)
        return {"final_report": report}
