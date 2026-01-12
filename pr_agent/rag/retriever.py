"""Retriever module for the PR Agent.

Handles retrieval of relevant code contexts using ChromaDB and semantic search.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("RAG")

# Robust Path to DB
# Using pathlib for better path manipulation across OS
ROOT_DIR = Path(__file__).parent.parent.parent
DB_DIR = str(ROOT_DIR / "chroma_db")


class ChromaRetriever:
    """Retriever using ChromaDB + Semantic Search."""

    def __init__(
        self,
        persist_path: str = DB_DIR,
        model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "main",
    ):
        """Initialize the ChromaRetriever.

        Args:
            persist_path: Directory for ChromaDB persistence.
            model_name: Name of the embedding model to use.
            collection_name: Name of the collection to use.
        """
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embedder = SentenceTransformer(model_name)

    def index_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """Indexes metadata-rich chunks into ChromaDB.

        Args:
            chunks: A list of logic chunks to index.
        """
        if not chunks:
            return

        ids = [f"{c['id']}_{i}" for i, c in enumerate(chunks)]
        documents = [c["content"] for c in chunks]
        metadatas = [
            {
                "name": c["id"],
                "filepath": c.get("filepath", "unknown"),
                "type": c["type"],
                "signature": c.get("signature", ""),
                "docstring": c.get("docstring", ""),
                "source": c["content"],
            }
            for c in chunks
        ]

        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)

    def retrieve_external_context(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Finds code in the index that is semantically similar to the query.

        Args:
            query: The search query.
            k: Number of results to return.

        Returns:
            A list of dictionary results containing metadata and source.
        """
        try:
            embedding = self.embedder.encode([query]).tolist()

            results = self.collection.query(query_embeddings=embedding, n_results=k)

            found = []
            if results and results.get("documents"):
                # Handle possible missing keys or empty lists safely
                docs = results.get("documents", [[]])[0]
                metas = results.get("metadatas", [[]])[0]

                if not docs or not metas:
                    return []

                for i in range(len(docs)):
                    meta = metas[i]
                    found.append(
                        {
                            "id": meta["name"],
                            "filepath": meta.get("filepath", "unknown"),
                            "type": "external_context",
                            "content": (
                                f"File: {meta['filepath']}\n"
                                f"Signature: {meta['signature']}\n"
                                f"Docstring: {meta['docstring']}\n"
                                f"Source:\n{meta['source']}"
                            ),
                        }
                    )
            return found
        except Exception as e:
            logger.warning(f"Chroma Retrieval failed: {e}")
            return []

    def list_collections(self) -> List[str]:
        """Returns a list of all available collection names."""
        try:
            return [c.name for c in self.client.list_collections()]
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

    def delete_collection(self, collection_name: str) -> bool:
        """Deletes a collection by name."""
        try:
            self.client.delete_collection(collection_name)
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection {collection_name}: {e}")
            return False
