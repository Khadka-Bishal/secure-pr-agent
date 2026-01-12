"""Secure PR Agent CLI.

Main entry point for the PR Agent application.
"""

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

from pr_agent.agent import PullRequestAgent
from pr_agent.git import GitHubClient
from pr_agent.llm import LLMClient
from pr_agent.rag import index_codebase
from pr_agent.utils.logger import setup_observability


def get_collection_name(target_path: str) -> str:
    """Infers collection name from a target path or URL.

    Args:
        target_path: The filesystem path or git URL.

    Returns:
        A normalized collection name.
    """
    if target_path in [".", "./"]:
        return os.path.basename(os.getcwd())

    if target_path.startswith(("http", "git@")):
        # e.g. https://github.com/owner/repo -> repo
        return target_path.rstrip("/").split("/")[-1].replace(".git", "")

    # Local path: /path/to/foo -> foo
    return os.path.basename(os.path.normpath(target_path))


def main() -> None:
    """Main execution function."""
    load_dotenv()
    setup_observability()

    parser = argparse.ArgumentParser(description="Secure PR Agent CLI")
    parser.add_argument("--pr", help="GitHub PR URL")
    parser.add_argument(
        "--provider", default="openai", help="LLM Provider (openai|ollama)"
    )
    parser.add_argument("--index", help="Path to local codebase to index")
    parser.add_argument(
        "--collection",
        default="main",
        help="Name of the vector collection (e.g. repo name)",
    )
    parser.add_argument(
        "--list-collections",
        action="store_true",
        help="List all available Indexes/Collections",
    )
    parser.add_argument("--delete-collection", help="Delete a specific collection")
    parser.add_argument(
        "--rereview", action="store_true", help="Force re-review even if already processed"
    )

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Mode: Delete Collection
    if args.delete_collection:
        from pr_agent.rag.retriever import ChromaRetriever

        try:
            retriever = ChromaRetriever()
            if retriever.delete_collection(args.delete_collection):
                print(f"Collection '{args.delete_collection}' deleted.")
                sys.exit(0)
            else:
                print(
                    f"Failed to delete collection '{args.delete_collection}' "
                    "(Does it exist?)."
                )
                sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    # Mode: List Collections
    if args.list_collections:
        from pr_agent.rag.retriever import ChromaRetriever

        try:
            retriever = ChromaRetriever()
            collections = retriever.list_collections()
            print("\nAvailable Collections:")
            for c in collections:
                print(f" - {c}")
            print("")
            sys.exit(0)
        except Exception as e:
            print(f"Error listing collections: {e}")
            sys.exit(1)

    # Mode: Indexing
    if args.index:
        if args.collection == "main":
            args.collection = get_collection_name(args.index)
            print(f"Auto-detected collection name: '{args.collection}'")

        print(f"Starting Indexing for {args.index}...")
        try:
            index_codebase(args.index, collection_name=args.collection)
            sys.exit(0)
        except Exception as e:
            print(f"Indexing Failed: {e}")
            sys.exit(1)

    # Mode: PR Review
    if not args.pr:
        print("Usage error: Please provide --pr <URL> OR --index <Path>")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"PR AGENT: Security Review")
    print(f"Target:     {args.pr}")
    print(f"Provider:   {args.provider}")
    print(f"Collection: {args.collection}")
    print("=" * 60 + "\n")

    try:
        agent = PullRequestAgent(
            llm=LLMClient(provider=args.provider),
            github=GitHubClient(),
            collection_name=args.collection,
            force=args.rereview,
        )
        agent.run(args.pr)
    except Exception as e:
        print(f"Fatal Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
