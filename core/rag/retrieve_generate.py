from pathlib import Path

from .config import TOP_K
from .base_run import parse_topics, append_block
from .retriever import clip_text
from .answerer import PROMPT_RETRIEVE_GENERATE


def run_retrieve_generate(
    topics_path: Path,
    log_path: Path,
    retriever,
    answerer,
    SegmentModel,
    topk: int = TOP_K,
    limit=None,  # iterate full file by default
):
    topics = parse_topics(topics_path, limit=limit)
    log_path.write_text("", encoding="utf-8")

    # Required: log prompt(s) once at the top
    append_block(log_path, "PROMPT_USED (Answer Generation)\n" + PROMPT_RETRIEVE_GENERATE + "\n\n")

    for i, (qid, qtext) in enumerate(topics, start=1):
        print(f"{i:02d}. {qtext}")

        # Retrieve using ORIGINAL query (Q3 requirement)
        scores, rows = retriever.search(qtext, topk)

        valid_rows = [r for r in rows if r != -1]
        segs = SegmentModel.objects.filter(row_id__in=valid_rows).only("row_id", "doc_id", "text")
        seg_map = {s.row_id: s for s in segs}

        # Build passages list in rank order (skip missing)
        passages = []
        for r in rows:
            seg = seg_map.get(r)
            if r != -1 and seg is not None:
                passages.append((seg.doc_id, seg.text))

        # Generate a single answer (even if passages empty)
        answer = answerer.answer(qtext, passages)

        block = []
        block.append("=" * 80 + "\n")
        block.append(f"Query {i:02d}\n")
        block.append(f"Query ID  : {qid}\n")
        block.append(f"Query Text: {qtext}\n")
        block.append("-" * 80 + "\n")

        for rank, (r, sc) in enumerate(zip(rows, scores), start=1):
            seg = seg_map.get(r)
            if r == -1 or seg is None:
                block.append(f"  [{rank}] <no result>\n\n")
                continue
            block.append(f"  [{rank}] docID: {seg.doc_id}\n")
            block.append(f"      score: {sc}\n")
            # Full text or clipped—your choice. If you want full text, use seg.text directly.
            block.append(f"      text : {clip_text(seg.text)}\n\n")

        block.append("Generated Response:\n")
        block.append(answer.strip() + "\n\n")

        append_block(log_path, "".join(block))
