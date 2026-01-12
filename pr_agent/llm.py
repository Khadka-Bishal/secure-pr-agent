"""LLM Client module.

Unified interface for OpenAI and Ollama using litellm.
"""

import logging
import os
import textwrap
from typing import Dict, List, Optional

from litellm import completion

from pr_agent.utils.logger import TOKEN_USAGE, get_tracer

tracer = get_tracer()
logger = logging.getLogger("LLMClient")


class LLMClient:
    """Abstraction layer for Large Language Model interactions.

    Supports failover and multi-provider configurations (OpenAI/Ollama).
    """

    def __init__(
        self, provider: str = "openai", model: str = "gpt-4o", temperature: float = 0.0
    ):
        """Initialize the LLM client."""
        self.provider = provider
        self.model = model
        self.temperature = temperature

        # Enforce smart defaults for local deployment
        if provider == "ollama" and model in ["gpt-4o", "default"]:
            self.model = "qwen2.5-coder:7b"
            logger.info(f"Using prioritized local model: {self.model}")

        if provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            logger.warning("OPENAI_API_KEY is not set.")

    def chat(self, messages: List[Dict[str, str]], json_output: bool = False) -> str:
        """Executes a chat completion request with observability instrumentation.

        Args:
            messages: List of message dicts (role, content).
            json_output: Whether to force JSON schema in response.

        Returns:
            The raw content string from the LLM.
        """
        with tracer.start_as_current_span("llm_call") as span:
            span.set_attribute("llm.provider", self.provider)
            span.set_attribute("llm.model", self.model)

            # Normalize model name for LiteLLM
            model_name = (
                f"ollama/{self.model}" if self.provider == "ollama" else self.model
            )

            try:
                response = completion(
                    model=model_name,
                    messages=messages,
                    temperature=self.temperature,
                    response_format={"type": "json_object"} if json_output else None,
                )

                content = response.choices[0].message.content

                # Metrics
                usage = response.usage
                TOKEN_USAGE.labels(model=self.model, type="prompt").inc(
                    usage.prompt_tokens
                )
                TOKEN_USAGE.labels(model=self.model, type="completion").inc(
                    usage.completion_tokens
                )

                return content

            except Exception as e:
                logger.error(f"LLM Interaction Failed: {e}")
                if self.provider == "ollama":
                    raise RuntimeError(
                        f"Ollama failure with model '{self.model}'.Verify 'ollama serve' is running."
                    ) from e

                span.record_exception(e)
                raise e

    def review_code_chunk(self, code: str, language: str) -> str:
        """Specialized method for code analysis tasks."""
        prompt = textwrap.dedent(f"""
            Act as a Principal Security Engineer. 
            Goal: Identify CRITICAL bugs, security vulnerabilities, or logic errors in the code below.
            
            Rules:
            1. "Negative Reporting Only": If you find NO issues, return the string "PASS" and nothing else.
            2. NO "Code Quality" or "Nitpicks" unless they cause a bug.
            3. NO Markdown Headers (e.g. # Findings). Use indented bullet points only.
            4. Be Extremely Concise. No fluff.
            
            Code:
            ```{language}
            {code}
            ```
        """)
        return self.chat(
            [
                {
                    "role": "system",
                    "content": "You represent a strict security audit logic gate. No fluff.",
                },
                {"role": "user", "content": prompt},
            ]
        )

    def generate_diagram(self, diff_content: str) -> str:
        """Generates a Mermaid Sequence Diagram to visualize the PR logic flow."""
        prompt = textwrap.dedent(f"""
            Goal: Create a Mermaid Sequence Diagram to visualize the logic flow of these code changes.
            
            Diff Summary:
            {diff_content[:4000]}
            
            Instructions:
            1. Use `sequenceDiagram`.
            2. Focus on the high-level flow (User -> Service -> DB).
            3. Do NOT include implementation details, just the contract/flow.
            4. Return ONLY the mermaid code block.
        """)
        return self.chat(
            [
                {
                    "role": "system",
                    "content": "You are a technical documenter. Return only mermaid code.",
                },
                {"role": "user", "content": prompt},
            ]
        )

    def fix_diagram_syntax(self, code: str, error_msg: str) -> str:
        """Repairs broken Mermaid code based on a specific Linter error."""
        prompt = textwrap.dedent(f"""
            Act as a Code Fixer.
            The following Mermaid code failed validation.
            
            Error: "{error_msg}"
            
            Broken Code:
            {code}
            
            Task:
            Fix the syntax error. Return ONLY the corrected mermaid code block.
        """)
        return self.chat(
            [
                {
                    "role": "system",
                    "content": "You are a code repair tool. Return only mermaid code.",
                },
                {"role": "user", "content": prompt},
            ]
        )
