import re

ANSWERISH_PREFIXES = (
    "this is", "it is", "the answer", "based on", "in summary", "overall",
    "good morning",  # optional, based on your observed weird expansions
)

def sanitize_modified_query(modified: str, original: str, max_words: int = 24) -> str:
    if not modified:
        return original

    s = modified.strip()

    # strip surrounding quotes/backticks
    s = s.strip('"\''"`")

    # collapse whitespace / newlines
    s = " ".join(s.split())

    # if it looks like an explanatory sentence, fall back
    s_low = s.lower()
    if s.endswith(".") or s_low.startswith(ANSWERISH_PREFIXES) or "answer from:" in s_low:
        return original

    # keep it short-ish like a query
    words = s.split()
    if len(words) > max_words:
        s = " ".join(words[:max_words])

    return s or original
