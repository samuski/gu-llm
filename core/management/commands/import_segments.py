from __future__ import annotations
from pathlib import Path
import json
import re

from django.core.management.base import BaseCommand
from core.models import Segment

DOCID_KEYS = ["docid", "doc_id", "docID", "document_id", "id"]
TEXT_KEYS  = ["segment", "text", "contents", "passage", "body"]

def _natnum(p: Path) -> int:
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else 0

class Command(BaseCommand):
    help = "Import all collection/*.jsonl into SQLite as Segment(row_id, doc_id, text)."

    def add_arguments(self, parser):
        parser.add_argument("--data-dir", dest="data_dir", default="./msmarco-reduced-trecrag")
        parser.add_argument("--batch", type=int, default=5000)

    def handle(self, *args, **opts):
        data_dir = Path(opts["data_dir"]).resolve()
        batch_size = int(opts["batch"])

        collection_dir = data_dir / "collection"
        files = sorted(collection_dir.glob("*.jsonl"), key=_natnum)
        if not files:
            raise SystemExit(f"No jsonl files found under {collection_dir}")

        # detect keys from first file
        sample = None
        with files[0].open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    sample = json.loads(line)
                    break
        if sample is None:
            raise SystemExit(f"{files[0]} is empty")

        docid_key = next((k for k in DOCID_KEYS if k in sample), None)
        text_key  = next((k for k in TEXT_KEYS if k in sample), None)
        if not docid_key or not text_key:
            raise SystemExit(f"Can't detect keys. Found keys: {sorted(sample.keys())}")

        self.stdout.write(f"Detected keys: docid={docid_key}, text={text_key}")
        self.stdout.write(f"Importing {len(files)} shard(s) from {collection_dir}")
        self.stdout.write("Clearing existing Segment rows...")
        Segment.objects.all().delete()

        row_id = 0
        inserted = 0
        batch = []

        for shard in files:
            self.stdout.write(f"Reading {shard.name} ...")
            with shard.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)

                    docid = obj.get(docid_key)
                    text = obj.get(text_key)
                    if docid is None or text is None:
                        raise SystemExit(f"Missing keys at global row_id={row_id} in {shard.name}")

                    batch.append(Segment(row_id=row_id, doc_id=str(docid), text=str(text)))
                    row_id += 1

                    if len(batch) >= batch_size:
                        Segment.objects.bulk_create(batch, batch_size=batch_size)
                        inserted += len(batch)
                        batch.clear()
                        self.stdout.write(f"Inserted {inserted}...")

        if batch:
            Segment.objects.bulk_create(batch, batch_size=batch_size)
            inserted += len(batch)

        self.stdout.write(self.style.SUCCESS(
            f"Done. Inserted {inserted} segments (row_id 0..{row_id-1})."
        ))
