import re

JUNK_PATTERNS = [
    r"\bAnswer from:\b",
    r"\bShow answers\b",
    r"\bQuora\b",
    r"\bBuy\s+Find\s+launch\b",
]

def is_junky(text: str) -> bool:
    t = (text or "")
    t_low = t.lower()
    for pat in JUNK_PATTERNS:
        if re.search(pat, t, flags=re.I):
            return True
    # super short “nav” stubs
    if len(t_low.strip()) < 40:
        return True
    return False
