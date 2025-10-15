# src/rag_store.py
from __future__ import annotations
import os, json, re
from pathlib import Path
from typing import List, Dict
import numpy as np
from openai import OpenAI

# Project paths
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ART_DIR = DATA_DIR / "articles"

# Streaming index (written by scripts/build_index_streaming.py)
INDEX_BIN   = DATA_DIR / "articles_index.bin"        # raw float32, appended
INDEX_META  = DATA_DIR / "articles_meta.jsonl"       # 1 JSON per row
INDEX_SHAPE = DATA_DIR / "articles_index.shape.json" # {"dim": 1536}

# Legacy monolithic fallback (if you ever had these)
INDEX_NPY  = DATA_DIR / "articles_index.npy"   # shape [N, D], normalized
INDEX_JSON = DATA_DIR / "articles_meta.json"   # [{"path":..., "preview":...}, ...]

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

# -------------------- OpenAI client --------------------
def _client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env or export it in the shell."
        )
    return OpenAI(api_key=key)

def _embed_query(q: str) -> np.ndarray:
    resp = _client().embeddings.create(model=EMBED_MODEL, input=[q])
    v = np.array(resp.data[0].embedding, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-12)

# -------------------- Streaming search --------------------
def _load_dim() -> int:
    return int(json.loads(INDEX_SHAPE.read_text())["dim"])

def _row_count(dim: int) -> int:
    nbytes = INDEX_BIN.stat().st_size
    return nbytes // (4 * dim)  # float32 bytes per value

def _read_meta_all() -> List[dict]:
    out = []
    with open(INDEX_META, "r", encoding="utf-8") as f:
        for line in f:
            out.append(json.loads(line))
    return out

def search(query: str, top_k: int = 8, scan_rows: int = 20000) -> List[Dict]:
    """
    Return top matches as [{score, path, preview, row}], highest score first.
    Works with the streaming on-disk index; falls back to legacy files.
    """
    # Streaming index path
    if INDEX_BIN.exists() and INDEX_META.exists() and INDEX_SHAPE.exists():
        dim = _load_dim()
        total = _row_count(dim)
        meta = _read_meta_all()
        if len(meta) != total:
            raise RuntimeError("Meta rows don't match vector rows; rebuild the index.")

        q = _embed_query(query)

        best_scores = np.full(top_k, -1e9, dtype=np.float32)
        best_rows   = np.full(top_k, -1, dtype=np.int64)

        # Windowed cosine search via memmap so we never load all vectors
        for start in range(0, total, scan_rows):
            end = min(total, start + scan_rows)
            count = end - start
            mm = np.memmap(INDEX_BIN, dtype=np.float32, mode="r",
                           offset=start * dim * 4, shape=(count, dim))
            sims = mm @ q  # cosine (vectors normalized during build)

            k = min(top_k, count)
            idx = np.argpartition(-sims, k - 1)[:k]
            cand_scores = sims[idx]
            cand_rows   = idx + start

            # Merge with running top-k
            all_scores = np.concatenate([best_scores, cand_scores])
            all_rows   = np.concatenate([best_rows,   cand_rows])
            top_idx    = np.argsort(-all_scores)[:top_k]
            best_scores, best_rows = all_scores[top_idx], all_rows[top_idx]

        results = []
        for s, r in zip(best_scores, best_rows):
            if r < 0:
                continue
            m = meta[int(r)]
            results.append({
                "score": float(s),
                "path":  m["path"],
                "preview": m.get("preview", ""),
                "row":   int(r),
            })
        return results

    # Legacy fallback (in-memory npy + json)
    if INDEX_NPY.exists() and INDEX_JSON.exists():
        vecs = np.load(INDEX_NPY, mmap_mode="r")
        meta = json.loads(INDEX_JSON.read_text())
        q = _embed_query(query)
        sims = vecs @ q
        idx = np.argsort(-sims)[:top_k]
        return [{
            "score": float(sims[i]),
            "path":  meta[i]["path"],
            "preview": meta[i].get("preview",""),
            "row":   int(i),
        } for i in idx]

    raise FileNotFoundError(
        "No article index found. Build it with: "
        "python scripts\\build_index_streaming.py --fresh"
    )

# -------------------- Public helper your app expects --------------------
def _strip_front_matter(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4 :]
    return text

def query_articles(query: str, k: int = 5, snippet_chars: int = 500) -> List[Dict]:
    """
    Compatibility layer expected by app.py.
    Returns a list of dicts: { 'score', 'path', 'snippet', 'source' }
    """
    hits = search(query, top_k=k)
    out: List[Dict] = []
    for h in hits:
        p = Path(h["path"])
        source = p.name
        snippet = h.get("preview", "")

        # If we want a slightly longer snippet, read the file safely
        try:
            if snippet_chars and snippet_chars > len(snippet) and p.exists():
                raw = p.read_text(encoding="utf-8", errors="ignore")
                body = _strip_front_matter(raw)
                # crude but effective: take first `snippet_chars` chars
                body = re.sub(r"\s+", " ", body).strip()
                snippet = body[:snippet_chars]
        except Exception:
            pass

        out.append({
            "score":   float(h["score"]),
            "path":    str(p),
            "snippet": snippet,
            "source":  source,
        })
    return out

# -------------------- Old builder API (discourage at runtime) --------------------
def build_index_from_dir(*_args, **_kwargs):
    """
    Kept only so imports don't break. Prefer the streaming builder:
      python scripts\\build_index_streaming.py --fresh
    """
    raise NotImplementedError(
        "Use scripts/build_index_streaming.py (streaming index) instead."
    )
