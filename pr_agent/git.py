"""GitHub Client module.

Handles interactions with GitHub API and utility functions for repository cloning.
"""

import logging
import os
import shutil
import subprocess
import tempfile
from typing import List, Optional, Tuple

from github import Github
from github.PullRequest import PullRequest

from pr_agent.utils.logger import get_tracer

tracer = get_tracer()
logger = logging.getLogger("GitHubClient")


class GitHubClient:
    """Wrapper around PyGithub with tracing and caching support."""

    def __init__(self, token: Optional[str] = None):
        """Initialize the GitHub client.

        Args:
            token: GitHub API token. If None, reads from GITHUB_TOKEN env var.

        Raises:
            ValueError: If no token is provided.
        """
        self.token = token or os.getenv("GITHUB_TOKEN")
        if not self.token:
            raise ValueError("Environment variable GITHUB_TOKEN is required.")

        self.g = Github(self.token)
        self._processed_commits = set()

    def get_pr_diff(
        self, pr_url: str, force: bool = False
    ) -> Tuple[Optional[PullRequest], Optional[List[dict]]]:
        """Fetches PR message and diff files.

        Validates idempotency via commit SHA to avoid re-processing unless forced.
        """
        repo_name, pr_number = self._parse_pr_url(pr_url)

        with tracer.start_as_current_span("fetch_pr_diff") as span:
            repo = self.g.get_repo(repo_name)
            pr = repo.get_pull(pr_number)

            latest_commit = pr.head.sha

            # Idempotency check: look for our own previous comments
            if not force:
                comments = pr.get_issue_comments()
                for comment in comments:
                    if (
                        "PR AGENT REVIEW" in comment.body
                        and latest_commit in comment.body
                    ):
                        logger.info(
                            f"Skipping: Review for commit {latest_commit[:8]} "
                            "already exists. Use --rereview to override."
                        )
                        return None, None

            # Session-level idempotency
            if latest_commit in self._processed_commits and not force:
                logger.info(
                    f"Commit {latest_commit} already processed (Session Cache). Skipping."
                )
                span.set_attribute("skipped", True)
                return None, None

            self._processed_commits.add(latest_commit)

            files = []
            for file in pr.get_files():
                # Fetch full file content from the PR head commit
                full_content = None
                try:
                    if file.status != "removed":
                        file_content = repo.get_contents(
                            file.filename, ref=latest_commit
                        )
                        if hasattr(file_content, "decoded_content"):
                            full_content = file_content.decoded_content.decode("utf-8")
                except Exception as e:
                    logger.warning(
                        f"Could not fetch full content for {file.filename}: {e}"
                    )

                files.append(
                    {
                        "filename": file.filename,
                        "patch": file.patch,
                        "content": full_content,
                        "status": file.status,
                        "sha": latest_commit,
                    }
                )

            return pr, files

    def get_pr(self, pr_url: str) -> PullRequest:
        """Retrieves the PR object from GitHub."""
        repo_name, pr_number = self._parse_pr_url(pr_url)
        repo = self.g.get_repo(repo_name)
        return repo.get_pull(pr_number)

    def post_comment(self, pr_url: str, body: str) -> None:
        """Posts a comment to the specified PR."""
        try:
            pr = self.get_pr(pr_url)
            pr.create_issue_comment(body)
            logger.info(f"Successfully posted comment to {pr_url}")
        except Exception as e:
            logger.error(f"Failed to post comment: {e}")

    def _parse_pr_url(self, url: str) -> Tuple[str, int]:
        """Parses a GitHub PR URL into (repo_name, pr_number)."""
        # Expected format: https://github.com/owner/repo/pull/123
        parts = url.rstrip("/").split("/")
        if "pull" in parts:
            try:
                idx = parts.index("pull")
                # owner/repo are indices -2 and -1 relative to 'pull'
                repo = f"{parts[idx - 2]}/{parts[idx - 1]}"
                pr_num = int(parts[idx + 1])
                return repo, pr_num
            except (IndexError, ValueError):
                pass
        raise ValueError(f"Invalid PR URL: {url}")


def clone_repo(repo_url: str) -> str:
    """Clones a remote repository to a temporary directory."""
    temp_dir = tempfile.mkdtemp(prefix="pr_agent_repo_")
    logger.info(f"Cloning {repo_url} to {temp_dir}...")

    try:
        # Shallow clone for performance
        subprocess.check_call(
            ["git", "clone", "--depth", "1", repo_url, temp_dir],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return temp_dir
    except subprocess.CalledProcessError as e:
        logger.error(f"Clone failed: {e}")
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise e
