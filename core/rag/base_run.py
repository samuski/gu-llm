from pathlib import Path
from django.db.models import Model

from .config import TOP_K, LIMIT
from .retriever import clip_text

def parse_topics(path: Path, limit: int = LIMIT):
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "\t" in line:
                qid, qtext = line.split("\t", 1)
            else:
                parts = line.split(maxsplit=1)
                qid, qtext = parts[0], (parts[1] if len(parts) > 1 else "")
            out.append((qid.strip(), qtext.strip()))
            if limit is not None and len(out) >= limit:
              break
    return out

def append_block(log_path: Path, block: str):
    with log_path.open("a", encoding="utf-8") as f:
        f.write(block)
        f.flush()  # critical: write every iteration

def run_base(topics_path: Path, log_path: Path, retriever, SegmentModel: Model, topk: int = TOP_K, limit: int = LIMIT):
    topics = parse_topics(topics_path, limit=limit)
    log_path.write_text("", encoding="utf-8")  # clear at start

    for i, (qid, qtext) in enumerate(topics, start=1):
        print(f"{i:02d}. {qtext}")

        scores, rows = retriever.search(qtext, topk)

        # FAISS commonly returns row offsets; we map them through row_id
        valid_rows = [r for r in rows if r != -1]
        segs = SegmentModel.objects.filter(row_id__in=valid_rows).only("row_id", "doc_id", "text")
        seg_map = {s.row_id: s for s in segs}

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
            block.append(f"      text : {clip_text(seg.text)}\n\n")

        append_block(log_path, "".join(block))
