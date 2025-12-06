import math
import re
from collections import Counter

_WORD = re.compile(r"[a-z0-9]+")

def _tok(s: str):
    return _WORD.findall((s or "").lower())

def bm25_rerank(query: str, docs: list[str], k1: float = 1.5, b: float = 0.75) -> list[float]:
    q = _tok(query)
    doc_tokens = [_tok(d) for d in docs]
    N = len(docs)
    if N == 0:
        return []

    # doc freq
    df = Counter()
    for toks in doc_tokens:
        for w in set(toks):
            df[w] += 1

    # idf
    idf = {w: math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5)) for w in df}

    lens = [len(toks) for toks in doc_tokens]
    avgdl = (sum(lens) / N) if N else 1.0

    scores = []
    for toks, dl in zip(doc_tokens, lens):
        tf = Counter(toks)
        s = 0.0
        for w in q:
            if w not in tf:
                continue
            denom = tf[w] + k1 * (1 - b + b * (dl / avgdl))
            s += idf.get(w, 0.0) * (tf[w] * (k1 + 1) / (denom if denom else 1.0))
        scores.append(s)
    return scores
