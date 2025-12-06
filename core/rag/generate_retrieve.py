from pathlib import Path

from .config import TOP_K, LIMIT
from .retriever import clip_text
from .base_run import parse_topics, append_block
from .query_expander import PROMPT_QUERY_EXPAND


def run_generate_retrieve(
    topics_path: Path,
    log_path: Path,
    retriever,
    expander,
    SegmentModel,
    topk: int = TOP_K,
    limit: int = LIMIT,
):
    topics = parse_topics(topics_path, limit=limit)
    log_path.write_text("", encoding="utf-8")

    # Log prompt(s) once at the top (required)
    append_block(log_path, "PROMPT_USED (Query Expansion)\n" + PROMPT_QUERY_EXPAND + "\n\n")

    for i, (qid, qtext) in enumerate(topics, start=1):
        print(f"{i:02d}. {qtext}")

        modified = expander.expand(qtext)
        scores, rows = retriever.search(modified, topk)

        valid_rows = [r for r in rows if r != -1]
        segs = SegmentModel.objects.filter(row_id__in=valid_rows).only("row_id", "doc_id", "text")
        seg_map = {s.row_id: s for s in segs}

        block = []
        block.append("=" * 80 + "\n")
        block.append(f"Query {i:02d}\n")
        block.append(f"Query ID           : {qid}\n")
        block.append(f"Query Text         : {qtext}\n")
        block.append(f"Modified Query Text: {modified}\n")
        block.append("-" * 80 + "\n")

        for rank, (r, sc) in enumerate(zip(rows, scores), start=1):
            seg = seg_map.get(r)
            if r == -1 or seg is None:
                block.append(f"  [{rank}] <no result>\n\n")
                continue
            block.append(f"  [{rank}] docID: {seg.doc_id}\n")
            block.append(f"      score: {sc}\n")
            block.append(f"      text : {clip_text(seg.text)}\n\n")

        append_block(log_path, "".join(block))
