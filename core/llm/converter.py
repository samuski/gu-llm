import os, re, json, argparse, pathlib
from glob import glob

def clean_text(t: str) -> str:
    if not isinstance(t, str): return ""
    t = t.replace("\u00a0"," ").replace("\t"," ").strip()
    t = re.sub(r"\[value_[^\]]+\]", "<VALUE>", t)  # normalize placeholders
    t = re.sub(r"\s+", " ", t)
    return t

def dialog_to_messages_turnlist(turns):
    """Generic turns -> messages (alternating user/assistant)."""
    msgs = []
    for t in turns:
        txt = clean_text(t)
        if not txt: continue
        # naive alternation starting with user
        role = "user" if len(msgs) % 2 == 0 else "assistant"
        msgs.append({"role": role, "content": txt})
    # enforce alternation (merge dup roles)
    out = []
    for m in msgs:
        if out and out[-1]["role"] == m["role"]:
            out[-1]["content"] = (out[-1]["content"] + "\n" + m["content"]).strip()
        else:
            out.append(m)
    # drop leading assistant if any
    while out and out[0]["role"] != "user": out.pop(0)
    return out

def parse_v21(root):
    """MultiWOZ_2.1: data.json + train/val/test lists."""
    p = pathlib.Path(root) / "data" / "MultiWOZ_2.1"
    data_json = p / "data.json"
    if not data_json.exists(): return None
    lists = {}
    for name in ["trainListFile.json","valListFile.json","testListFile.json"]:
        fp = p / name
        lists[name] = set(json.load(open(fp, encoding="utf-8"))) if fp.exists() else set()
    data = json.load(open(data_json, encoding="utf-8"))
    buckets = {"train": [], "val": [], "test": []}
    for d_id, dlg in data.items():
        turns = dlg.get("log") or []
        # v2.1 stores alternating 'text' fields; harvest in order
        texts = [x.get("text","") for x in turns if isinstance(x, dict)]
        msgs = dialog_to_messages_turnlist(texts)
        if len(msgs) < 2: continue
        bucket = "train"
        if d_id in lists["valListFile.json"]:  bucket = "val"
        if d_id in lists["testListFile.json"]: bucket = "test"
        buckets[bucket].append({"messages":[
            {"role":"system","content":"You are a concise, helpful travel assistant."},
            *msgs
        ]})
    return buckets

def parse_v22(root):
    """MultiWOZ_2.2: SGD-style folders train/dev/test with dialogues_*.json."""
    p = pathlib.Path(root) / "data" / "MultiWOZ_2.2"
    if not p.exists(): return None
    def collect(split_dir):
        out = []
        for fp in glob(str(split_dir / "dialogues_*.json")):
            arr = json.load(open(fp, encoding="utf-8"))
            # each item has turns with speaker USER/SYSTEM and 'utterance'
            for dlg in arr:
                msgs = []
                for t in dlg.get("turns", []):
                    spk = t.get("speaker","").lower()
                    utt = clean_text(t.get("utterance",""))
                    if not utt: continue
                    role = "user" if spk == "user" else ("assistant" if spk == "system" else None)
                    if role is None: continue
                    msgs.append({"role": role, "content": utt})
                # normalize (start with user, merge dup roles)
                if not msgs: continue
                while msgs and msgs[0]["role"] != "user": msgs.pop(0)
                norm = []
                for m in msgs:
                    if norm and norm[-1]["role"] == m["role"]:
                        norm[-1]["content"] = (norm[-1]["content"] + "\n" + m["content"]).strip()
                    else:
                        norm.append(m)
                if len(norm) < 2: continue
                out.append({"messages":[
                    {"role":"system","content":"You are a concise, helpful task-oriented assistant for UK travel and bookings."},
                    *norm
                ]})
        return out
    buckets = {
        "train": collect(p / "train"),
        "val":   collect(p / "dev"),
        "test":  collect(p / "test"),
    }
    return buckets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--multiwoz_dir", default="./multiwoz", help="path to the cloned repo")
    ap.add_argument("--out_dir", default="./data", help="where to write *.jsonl")
    ap.add_argument("--max_chars", type=int, default=12000, help="rough length guard (true token limit enforced later)")
    args = ap.parse_args()

    buckets = parse_v22(args.multiwoz_dir) or parse_v21(args.multiwoz_dir)
    if buckets is None:
        raise SystemExit("Could not detect MultiWOZ 2.1 or 2.2 structure. Check paths and unzip archives.")

    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split, rows in buckets.items():
        kept = 0; skipped = 0
        with open(out_dir / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                L = sum(len(m["content"]) for m in r["messages"])
                if L > args.max_chars: 
                    skipped += 1
                    continue
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                kept += 1
        stats[split] = {"kept": kept, "skipped": skipped}
    print("[convert] done. stats:", stats)

if __name__ == "__main__":
    main()
