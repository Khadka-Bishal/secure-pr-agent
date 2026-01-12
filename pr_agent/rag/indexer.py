"""Codebase Indexer module.

Scans the repository and indexes code chunks into ChromaDB.
"""

import logging
import os
import shutil
from typing import List, Optional

from pr_agent.git import clone_repo
from .chunker import CodeChunker
from .retriever import ChromaRetriever

logger = logging.getLogger("Indexer")

# Directories to exclude from indexing
EXCLUDED_DIRS = {".git", "__pycache__", "venv", "node_modules", "dist", "build"}
# Supported file extensions
SUPPORTED_EXTENSIONS = (".py", ".js", ".ts", ".go", ".rs", ".java")


def index_codebase(root_dir: str, collection_name: str = "main") -> None:
    """Walks the codebase, chunks files, and indexes them into ChromaDB.

    Args:
        root_dir: The path to the local directory or a git URL.
        collection_name: The name of the ChromaDB collection.
    """
    is_temp = False

    # Handle remote repositories
    if root_dir.startswith(("http://", "https://", "git@")):
        try:
            logger.info(f"Detected remote repository: {root_dir}")
            root_dir = clone_repo(root_dir)
            is_temp = True
        except Exception as e:
            logger.error(f"Failed to clone remote repo: {e}")
            return

    try:
        chunker = CodeChunker()
        embedding_model = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        retriever = ChromaRetriever(
            model_name=embedding_model, collection_name=collection_name
        )

        logger.info(
            f"Indexing {root_dir} into collection '{collection_name}' "
            f"using {embedding_model}..."
        )

        all_chunks = []

        for root, dirs, _ in os.walk(root_dir):
            # Prune excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]

            for file in os.listdir(root):
                if file.endswith(SUPPORTED_EXTENSIONS):
                    filepath = os.path.join(root, file)
                    if not os.path.isfile(filepath):
                        continue
                        
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            content = f.read()

                        chunks = chunker.chunk_code(file, content)
                        rel_path = os.path.relpath(filepath, root_dir)

                        for chunk in chunks:
                            chunk["filepath"] = rel_path

                        all_chunks.extend(chunks)
                    except Exception as e:
                        logger.warning(f"Failed to read {filepath}: {e}")

        if all_chunks:
            logger.info(f"Found {len(all_chunks)} chunks. Indexing...")
            retriever.index_documents(all_chunks)
            logger.info("Indexing Complete.")
        else:
            logger.info("No supported code files found to index.")

    finally:
        if is_temp and os.path.exists(root_dir):
            logger.info(f"Cleaning up temporary directory: {root_dir}")
            shutil.rmtree(root_dir)
