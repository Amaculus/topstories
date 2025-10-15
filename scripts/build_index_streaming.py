# scripts/build_index_streaming.py
from __future__ import annotations
import argparse, os, json, time, math, re, hashlib
from pathlib import Path
from typing import Iterable, List, Tuple
import numpy as np

# ---- project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ART_DIR = DATA_DIR / "articles"
INDEX_BIN = DATA_DIR / "articles_index.bin"       # raw float32 vectors, appended
INDEX_META = DATA_DIR / "articles_meta.jsonl"     # one json per row (path, chunk, text preview)
INDEX_SHAPE = DATA_DIR / "articles_index.shape.json"  # {"dim": 1536}

# ---- OpenAI embeddings
import os
from openai import OpenAI
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

def _client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_batch(texts: List[str]) -> np.ndarray:
    resp = _client().embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = np.array([d.embedding for d in resp.data], dtype=np.float32)
    # L2-normalize so dot == cosine
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
    return vecs / norms

# ---- tiny chunker (char-based, stable & fast)
def chunk_text(s: str, size: int = 1200, overlap: int = 200) -> Iterable[Tuple[int, str]]:
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return []
    i, n = 0, len(s)
    out = []
    while i < n:
        j = min(n, i + size)
        out.append((i, s[i:j]))
        if j >= n:
            break
        i = max(0, j - overlap)
    return out

def read_article(path: Path) -> str:
    t = path.read_text(encoding="utf-8", errors="ignore")
    # strip minimal YAML front matter if present
    if t.startswith("---"):
        end = t.find("\n---", 3)
        if end != -1:
            t = t[end + 4 :]
    return t

def append_vectors(bin_path: Path, arr: np.ndarray):
    # append raw float32 bytes to .bin (no header)
    with open(bin_path, "ab") as f:
        arr.astype(np.float32).tofile(f)

def write_meta_line(meta_path: Path, obj: dict):
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def build_streaming(
    articles_dir: Path,
    chars_per_chunk: int = 1200,
    overlap: int = 200,
    batch: int = 4,
    fresh: bool = False,
):
    articles = sorted(articles_dir.glob("*.md"))
    if fresh:
        for p in (INDEX_BIN, INDEX_META, INDEX_SHAPE):
            if p.exists():
                p.unlink()

    total_rows = 0
    dim_known = None

    for p in articles:
        raw = read_article(p)
        chunks = chunk_text(raw, size=chars_per_chunk, overlap=overlap)
        if not chunks:
            continue

        # embed in tiny batches
        buf_text, buf_meta = [], []
        for start, text in chunks:
            buf_text.append(text)
            buf_meta.append({"path": str(p), "start": start, "preview": text[:200]})
            if len(buf_text) >= batch:
                vecs = embed_batch(buf_text)
                append_vectors(INDEX_BIN, vecs)
                for m in buf_meta:
                    write_meta_line(INDEX_META, m)
                total_rows += vecs.shape[0]
                dim_known = vecs.shape[1]
                buf_text, buf_meta = [], []

        if buf_text:
            vecs = embed_batch(buf_text)
            append_vectors(INDEX_BIN, vecs)
            for m in buf_meta:
                write_meta_line(INDEX_META, m)
            total_rows += vecs.shape[0]
            dim_known = vecs.shape[1]

        print(f"SAVED: {p.name}  (+{len(chunks)} chunks)  total_rows={total_rows}")

    # record dimension so we can memory-map later
    if dim_known is None:
        raise RuntimeError("No vectors were written; check your articles directory.")
    INDEX_SHAPE.write_text(json.dumps({"dim": int(dim_known)}, indent=2))
    print(f"Done. rows={total_rows} dim={dim_known}")
    print(f"- {INDEX_BIN}")
    print(f"- {INDEX_META}")
    print(f"- {INDEX_SHAPE}")

def main():
    ap = argparse.ArgumentParser(description="Streaming, low-memory RAG index builder")
    ap.add_argument("--dir", default=str(ART_DIR))
    ap.add_argument("--chunk", type=int, default=1200)
    ap.add_argument("--overlap", type=int, default=200)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--fresh", action="store_true", help="remove existing index files")
    args = ap.parse_args()

    build_streaming(
        Path(args.dir),
        chars_per_chunk=args.chunk,
        overlap=args.overlap,
        batch=args.batch,
        fresh=args.fresh,
    )

if __name__ == "__main__":
    main()
