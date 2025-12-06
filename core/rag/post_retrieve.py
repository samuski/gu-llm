# core/rag/post_retrieve.py

from dataclasses import dataclass
import re

@dataclass
class Hit:
    row_id: int
    score: float       # keep this as FAISS score for logging consistency
    seg: object        # Segment model instance

_word = re.compile(r"[a-z0-9]+")

def _tokens(s: str) -> set[str]:
    return set(_word.findall((s or "").lower()))

def _rerank_overlap(query: str, text: str) -> float:
    q = _tokens(query)
    t = _tokens(text)
    if not q or not t:
        return 0.0
    return len(q & t) / (len(q) ** 0.5)

def _looks_junky(text: str) -> bool:
    t = (text or "").strip()
    if len(t) < 40:
        return True
    # super noisy “Show answers Answer from:” style
    low = t.lower()
    if "answer from:" in low and "show answers" in low:
        return True
    return False

def select_topk(query: str, rows, scores, seg_map, topk: int = 3):
    # 1) build base candidates in FAISS rank order
    base = []
    for r, sc in zip(rows, scores):
        if r == -1:
            continue
        seg = seg_map.get(r)
        if seg is None:
            continue
        base.append(Hit(row_id=r, score=float(sc), seg=seg))

    if not base:
        return []

    # 2) filter (soft): keep only non-junky if possible
    filtered = [h for h in base if not _looks_junky(h.seg.text)]
    pool = filtered if filtered else base  # fallback if filter kills everything

    # 3) rerank within pool (optional), but do NOT lose FAISS score
    pool_sorted = sorted(pool, key=lambda h: _rerank_overlap(query, h.seg.text), reverse=True)

    # 4) final fallback: if rerank produced nothing (shouldn’t), use base
    final = pool_sorted if pool_sorted else base

    return final[:topk]
