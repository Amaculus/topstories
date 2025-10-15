# scripts/csv_to_jsonl_evergreen.py
import csv, json, sys
from pathlib import Path

def convert(csv_path: str, out_jsonl: str):
    src = Path(csv_path)
    rows = []
    with src.open("r", encoding="utf-8", newline="") as f:
        # Your sheet has no header row; columns are positional.
        rdr = csv.reader(f)
        for row in rdr:
            if not row:
                continue
            # Col indices from your file:
            # 0=url, 15=Title, 16=SEO Title, 17=Description
            url = (row[0] or "").strip() if len(row) > 0 else ""
            if not url.startswith("http"):
                continue
            title = ""
            if len(row) > 16 and row[16].strip():
                title = row[16].strip()
            elif len(row) > 15 and row[15].strip():
                title = row[15].strip()
            desc = (row[17].strip() if len(row) > 17 else "")
            anchors = []
            if title:
                anchors = [title, title.split("|")[0].strip()]
                # de-dupe while preserving order
                anchors = list(dict.fromkeys(a for a in anchors if a))

            rows.append({
                "url": url,
                "title": title,
                "anchors": anchors,
                "description": desc
            })

    with open(out_jsonl, "w", encoding="utf-8") as out:
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    csv_in = sys.argv[1] if len(sys.argv) > 1 else "data/evergreen.csv"
    jsonl_out = sys.argv[2] if len(sys.argv) > 2 else "data/evergreen.jsonl"
    convert(csv_in, jsonl_out)
    print(f"Converted {csv_in} -> {jsonl_out}")
