"""Utility modules for rate limiting and caching."""

from typing import Any


def extract_text_content(content: Any) -> str:
    """Extract string content from a LangChain message content.

    Handles both plain strings and list of text block dictionaries.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text" and "text" in block:
                    parts.append(block["text"])
                elif "text" in block:
                    parts.append(block["text"])
        return "".join(parts)
    return str(content) if content is not None else ""
