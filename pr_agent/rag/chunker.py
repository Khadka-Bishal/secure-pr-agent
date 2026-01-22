"""Code chunking module.

This module provides functionality to split source code into syntax-aware chunks
using AST for Python and heuristic parsing for other languages.
"""

import ast
import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger("RAG")


class CodeChunker:
    """Splits code into syntax-aware chunks."""

    def __init__(self):
        """Initialize the CodeChunker."""
        pass

    def chunk_code(
        self, filename: str, content: str, patch: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Chunks content based on file extension.

        Args:
            filename: The name of the file.
            content: The file content (full file or patch).
            patch: Optional patch/diff to identify changed lines.

        Returns:
            A list of dictionary chunks.
        """
        if filename.endswith(".py"):
            return self._chunk_python_ast(content, patch)
        elif filename.endswith(".go"):
            return self._chunk_brace_parsing(content, "go")
        elif filename.endswith((".js", ".ts", ".jsx", ".tsx")):
            return self._chunk_brace_parsing(content, "js")
        else:
            return self._chunk_simple(content)

    def _chunk_brace_parsing(self, content: str, language: str) -> List[Dict[str, str]]:
        """Chunks C-style languages (Go, JS, TS) using regex and brace counting."""
        chunks = []
        lines = content.splitlines()

        patterns = {
            "go": [
                r"^func\s+(\w+)",  # func Foo
                r"^func\s+\(.*\)\s+(\w+)",  # func (s *Struct) Foo
                r"^type\s+(\w+)\s+struct",  # type Foo struct
            ],
            "js": [
                r"^function\s+(\w+)",  # function foo
                r"^class\s+(\w+)",  # class Foo
                r"^(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?(?:function|\(.*?\)\s*=>)",  # const foo ...
            ],
        }
        # TypeScript uses JS patterns
        patterns["ts"] = patterns["js"]

        active_patterns = patterns.get(language, patterns["js"])

        current_chunk: List[str] = []
        current_id = "unknown"
        brace_balance = 0
        in_chunk = False

        for line in lines:
            if not in_chunk:
                match_found = False
                for p in active_patterns:
                    match = re.search(p, line.strip())
                    if match:
                        current_id = match.group(1)
                        in_chunk = True
                        match_found = True
                        break

                if match_found:
                    brace_balance = 0
                    current_chunk = []

            if in_chunk:
                current_chunk.append(line)
                brace_balance += line.count("{")
                brace_balance -= line.count("}")

                if brace_balance <= 0:
                    if current_chunk:
                        chunks.append(
                            {
                                "id": current_id,
                                "type": "function",
                                "content": "\n".join(current_chunk),
                                "docstring": f"Detected {language} block: {current_id}",
                            }
                        )
                    in_chunk = False
                    current_chunk = []

        if not chunks:
            return self._chunk_simple(content)

        return chunks

    def _chunk_python_ast(
        self, content: str, patch: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Uses Python's AST to extract functions and classes.

        If patch is provided, extracts changed line numbers and only returns
        chunks containing those changes.
        """
        chunks = []
        try:
            tree = ast.parse(content)
            changed_lines = self._extract_changed_lines(patch) if patch else None

            for node in ast.iter_child_nodes(tree):
                if isinstance(
                    node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
                ):
                    lines = content.splitlines()
                    # correct 0-indexed adjustment is needed, ast uses 1-based lineno
                    start = node.lineno - 1
                    end = node.end_lineno

                    if start < 0 or end is None:
                        continue

                    # If patch provided, only include chunks with changes
                    if changed_lines is not None:
                        node_lines = set(range(node.lineno, (end or node.lineno) + 1))
                        if not node_lines.intersection(changed_lines):
                            continue

                    source_segment = "\n".join(lines[start:end])

                    docstring = ast.get_docstring(node) or "No docstring"
                    chunk_type = (
                        "class" if isinstance(node, ast.ClassDef) else "function"
                    )

                    chunks.append(
                        {
                            "id": node.name,
                            "type": chunk_type,
                            "docstring": docstring,
                            "content": source_segment,
                        }
                    )
        except SyntaxError:
            # Fallback for invalid Python syntax
            return self._chunk_simple(content)

        return chunks if chunks else self._chunk_simple(content)

    def _extract_changed_lines(self, patch: str) -> set:
        """Extract line numbers that were added or modified from a patch.

        Args:
            patch: Git diff patch string.

        Returns:
            Set of line numbers (1-based) that were changed.
        """
        changed_lines = set()
        if not patch:
            return changed_lines

        current_line = 0
        for line in patch.split("\n"):
            # Parse hunk headers like @@ -15,6 +15,8 @@
            if line.startswith("@@"):
                try:
                    # Extract new file line numbers from +15,8 format
                    parts = line.split("+")
                    if len(parts) > 1:
                        nums = parts[1].split()[0].split(",")
                        current_line = int(nums[0])
                except (ValueError, IndexError):
                    continue
            elif line.startswith("+") and not line.startswith("+++"):
                # Added line
                changed_lines.add(current_line)
                current_line += 1
            elif not line.startswith("-"):
                # Context line (not removed)
                current_line += 1

        return changed_lines

    def _chunk_simple(self, content: str) -> List[Dict[str, str]]:
        """Fallback chunker for non-supported languages or failed parsing."""
        return [
            {
                "id": "whole_file",
                "type": "file",
                "content": content,
                "docstring": "No docstring available",
            }
        ]
