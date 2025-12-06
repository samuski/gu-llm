# core/rag/sanitize.py
import re

def sanitize_query_rewrite(s: str) -> str:
    """Normalize expander output into a single-line query-like string."""
    if not s:
        return ""
    s = s.strip()

    # remove surrounding quotes
    s = s.strip('"\'')

    # drop common wrappers
    s = re.sub(
        r"^(modified query text|modified query|rewritten query|rewrite|expanded query)\s*:\s*",
        "",
        s,
        flags=re.I,
    ).strip()

    # collapse whitespace/newlines
    s = " ".join(s.split())
    return s

def is_good_rewrite(s: str, min_words: int = 3, max_words: int = 32) -> bool:
    """Decide whether to trust the rewrite, else fallback to original query."""
    if not s:
        return False

    words = s.split()
    if len(words) < min_words or len(words) > max_words:
        return False

    # "answer-y" patterns
    low = s.lower()
    if s.endswith("."):
        return False
    if low.startswith(("it is", "this is", "the answer", "answer:", "explanation:")):
        return False
    if "according to" in low:
        return False

    return True
