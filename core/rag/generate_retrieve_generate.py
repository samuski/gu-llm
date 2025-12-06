from pathlib import Path

from .base_run import parse_topics, append_block
from .retriever import clip_text
from .query_expander import PROMPT_QUERY_EXPAND
from .answerer import PROMPT_RETRIEVE_GENERATE
from .post_retrieve import select_topk
from .sanitize import sanitize_query_rewrite, is_good_rewrite
from .config import CANDIDATE_K


def _get_hit_fields(hit):
    # Hit object / dataclass
    if hasattr(hit, "seg"):
        return getattr(hit, "seg"), getattr(hit, "score", None)
    if hasattr(hit, "segment"):
        return getattr(hit, "segment"), getattr(hit, "score", None)

    # dict
    if isinstance(hit, dict):
        seg = hit.get("seg") or hit.get("segment")
        return seg, hit.get("score")

    # tuple/list
    if isinstance(hit, (tuple, list)):
        if len(hit) == 3:
            _, sc, seg = hit
            return seg, sc
        if len(hit) == 2:
            sc, seg = hit
            return seg, sc

    return None, None


def run_generate_retrieve_generate(
    topics_path: Path,
    log_path: Path,
    retriever,
    expander,
    answerer,
    SegmentModel,
    topk: int = 3,
    limit=None,
):
    topics = parse_topics(topics_path, limit=limit)
    log_path.write_text("", encoding="utf-8")

    append_block(
        log_path,
        "PROMPT_USED (Query Expansion)\n" + PROMPT_QUERY_EXPAND + "\n\n"
        "PROMPT_USED (Answer Generation)\n" + PROMPT_RETRIEVE_GENERATE + "\n\n"
    )

    for i, (qid, qtext) in enumerate(topics, start=1):
        print(f"{i:02d}. {qtext}")

        modified_raw = expander.expand(qtext)
        modified_clean = sanitize_query_rewrite(modified_raw)

        query_used_for_retrieval = modified_clean if is_good_rewrite(modified_clean) else qtext

        scores, rows = retriever.search(query_used_for_retrieval, CANDIDATE_K)

        valid_rows = [r for r in rows if r != -1]
        segs = SegmentModel.objects.filter(row_id__in=valid_rows).only("row_id", "doc_id", "text")
        seg_map = {s.row_id: s for s in segs}

        found = sum(1 for r in valid_rows if r in seg_map)
        print("faiss rows min/max:", (min(valid_rows), max(valid_rows)) if valid_rows else None,
              "| found_text:", found, "/", len(valid_rows),
              "| db_max_row:", SegmentModel.objects.order_by("-row_id").first().row_id)

        top_hits = select_topk(query_used_for_retrieval, rows, scores, seg_map, topk=topk)
        # top_hits = []
        # for r, sc in zip(rows[:topk], scores[:topk]):
        #     seg = seg_map.get(r)
        #     top_hits.append((r, sc, seg))

        passages = []
        for hit in top_hits:
            seg, _ = _get_hit_fields(hit)
            if seg is not None:
                passages.append((seg.doc_id, seg.text))

        answer = answerer.answer(qtext, passages)

        block = []
        block.append("=" * 80 + "\n")
        block.append(f"Query {i:02d}\n")
        block.append(f"Query ID           : {qid}\n")
        block.append(f"Query Text         : {qtext}\n")
        block.append(f"Modified Query Text: {query_used_for_retrieval}\n")
        block.append("-" * 80 + "\n")

        for rank, hit in enumerate(top_hits, start=1):
            seg, sc = _get_hit_fields(hit)
            if seg is None:
                block.append(f"  [{rank}] <no result>\n\n")
                continue
            block.append(f"  [{rank}] docID: {seg.doc_id}\n")
            block.append(f"      score: {sc}\n")
            block.append(f"      text : {clip_text(seg.text)}\n\n")

        block.append("Generated Response:\n")
        block.append(answer.strip() + "\n\n")

        append_block(log_path, "".join(block))
