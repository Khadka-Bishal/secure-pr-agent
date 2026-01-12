import re
from typing import Tuple

# Valid Mermaid sequence diagram arrows.
VALID_ARROWS = {
    r"->", r"-->", 
    r"->>", r"-->>", 
    r"-x", r"--x", 
    r"-\)", r"--\)"
}

def validate_mermaid_structure(code: str) -> Tuple[bool, str]:
    """Validates Mermaid Sequence Diagram syntax.

    Args:
        code: The mermaid code block string.

    Returns:
        A tuple (is_valid, message).
    """
    lines = [line.strip() for line in code.strip().splitlines() if line.strip()]
    if not lines:
        return False, "Empty code block."

    # First non-comment line must be 'sequenceDiagram'
    header_found = False
    for line in lines:
        if line.startswith("%%"):
            continue
        if line == "sequenceDiagram":
            header_found = True
            break
        return False, f"First non-comment line must be 'sequenceDiagram', found: '{line}'"

    if not header_found:
        return False, "Missing 'sequenceDiagram' header."

    # Validate interaction lines
    for i, line in enumerate(lines):
        # Skip comments, header, and participant definitions
        if (
            line.startswith("%%")
            or line == "sequenceDiagram"
            or line.startswith(("participant ", "actor ", "autonumber"))
        ):
            continue

        # Check for interaction (contains colon)
        if ":" in line:
            interaction_part = line.split(":", 1)[0]

            # Notes are valid but don't need arrow validation
            if line.startswith("Note "):
                continue

            # Check if any valid arrow is present in the interaction part
            has_valid_arrow = any(
                re.search(arrow, interaction_part) for arrow in VALID_ARROWS
            )

            if not has_valid_arrow:
                # Heuristic: if it looks like an arrow but isn't valid
                if ">" in interaction_part or "-" in interaction_part:
                    return (
                        False,
                        f"Line {i+1}: Invalid arrow syntax in '{interaction_part}'. "
                        "Use ->, -->, ->>, -->>.",
                    )

    return True, "VALID"
