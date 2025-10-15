# scripts/ingest_gsc_top.py
from __future__ import annotations

"""
Ingest top URLs from a GSC export, fetch article text, save to data/articles/*.md
with lightweight YAML/JSON front matter. Optionally rebuild RAG index.

Usage (PowerShell):
  python scripts\ingest_gsc_top.py --csv data\gsc_top.csv \
      --prefix /legal-online-sports-betting/ --limit 60 --throttle 0.0 --no-index
"""

import os
import re
import json
import time
import hashlib
from pathlib import Path
from urllib.parse import urlparse, urlsplit, urlunsplit

import pandas as pd
import requests
from bs4 import BeautifulSoup

# --- Bootstrapping: make "src" importable if/when we build the index -----------------
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

# --- Locations ----------------------------------------------------------------------
DATA_DIR = ROOT / "data"
ART_DIR = DATA_DIR / "articles"
ART_DIR.mkdir(parents=True, exist_ok=True)

# --- Config knobs -------------------------------------------------------------------
VERBOSE = True  # set False to reduce logging
UA = (
    os.getenv("PTW_USER_AGENT")
    or "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)

# Try Trafilatura; fall back to requests + BeautifulSoup
try:
    import trafilatura
    HAVE_TRAFI, TRAFI_ERR = True, ""
except Exception as e:
    HAVE_TRAFI, TRAFI_ERR = False, str(e)


# ------------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------------
def _normalize_url(u: str) -> str:
    """Strip query/fragment; canonicalize trailing slash."""
    p = urlsplit(str(u))
    return urlunsplit((p.scheme, p.netloc, p.path.rstrip("/"), "", ""))


def _extract_bs(html: str) -> str:
    """Heuristic main-content extraction with BeautifulSoup."""
    soup = BeautifulSoup(html or "", "html.parser")
    node = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", attrs={"class": re.compile(r"(post|entry|content)")})
        or soup.body
    )
    ps = node.find_all("p") if node else soup.find_all("p")
    text = " ".join(p.get_text(" ", strip=True) for p in ps)
    return re.sub(r"\s+", " ", text).strip()


def load_gsc_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"GSC CSV not found: {path}")
    df = pd.read_csv(p)
    # try to normalize helpful numeric columns
    for want in ("clicks", "impressions", "ctr", "position"):
        for c in list(df.columns):
            if c.lower().strip() == want:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df


def pick_top_urls(df: pd.DataFrame, prefix: str, limit: int = 50) -> list[str]:
    # choose the column that holds URLs
    url_col = next(
        (c for c in df.columns if c.lower().strip() in ("page", "url", "page url")),
        df.columns[0],
    )
    d = df.copy()
    d[url_col] = d[url_col].astype(str)

    # filter by prefix (if provided)
    if prefix:
        d = d[d[url_col].str.contains(prefix, na=False)]

    # drop obvious paginated list pages like /page/2/
    d = d[~d[url_col].str.contains(r"/page/\d+/?$", na=False)]

    # canonicalize & dedup
    d["canon"] = d[url_col].map(_normalize_url)
    d = d.drop_duplicates(subset=["canon"])

    # scoring using clicks/impressions if present
    clicks = d.get("clicks", pd.Series(0, index=d.index)).astype(float)
    imps = d.get("impressions", pd.Series(0, index=d.index)).astype(float)
    d = d.assign(score=(clicks * 10) + imps).sort_values("score", ascending=False)

    return d["canon"].head(limit).tolist()


def fetch_main_text(url: str) -> tuple[str, dict, str]:
    """
    Return (text, meta, reason). 'reason' is '' when OK,
    else 'empty_after_bs', 'http_error:...', etc.
    """
    # 1) Trafilatura
    if HAVE_TRAFI:
        try:
            downloaded = trafilatura.fetch_url(url, no_ssl=True)
            text = trafilatura.extract(
                downloaded, include_formatting=False, include_links=False
            ) or ""
            if text:
                soup = BeautifulSoup(downloaded or "", "html.parser")
                meta = {
                    "title": (soup.title.string or "").strip() if soup.title else "",
                    "published": "",
                }
                for k in (
                    "article:published_time",
                    "og:updated_time",
                    "article:modified_time",
                    "date",
                    "pubdate",
                ):
                    el = soup.find("meta", {"property": k}) or soup.find(
                        "meta", {"name": k}
                    )
                    if el and el.get("content"):
                        meta["published"] = el["content"].strip()
                        break
                return text[:10000], meta, ""
        except Exception as e:
            if VERBOSE:
                print("trafilatura_failed:", type(e).__name__, url)

    # 2) Requests + BS fallback
    try:
        r = requests.get(
            url,
            headers={"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"},
            timeout=25,
            allow_redirects=True,
        )
        r.raise_for_status()
        text = _extract_bs(r.text)
        soup = BeautifulSoup(r.text, "html.parser")
        meta = {
            "title": (soup.title.string or "").strip() if soup.title else "",
            "published": "",
        }
        for k in (
            "article:published_time",
            "og:updated_time",
            "article:modified_time",
            "date",
            "pubdate",
        ):
            el = soup.find("meta", {"property": k}) or soup.find("meta", {"name": k})
            if el and el.get("content"):
                meta["published"] = el["content"].strip()
                break
        if text:
            return text[:10000], meta, ""
        return "", {}, "empty_after_bs"
    except Exception as e:
        return "", {}, f"http_error:{type(e).__name__}"


def should_keep(text: str) -> bool:
    # Only drop truly empty/garbage pages
    return len(text or "") >= 120


def save_markdown(url: str, text: str, meta: dict) -> str:
    host = urlparse(url).netloc.replace("www.", "")
    hsh = hashlib.sha1(url.encode()).hexdigest()[:10]
    title_slug = re.sub(r"[^a-z0-9\-]+", "-", (meta.get("title") or host).lower()).strip(
        "-"
    )
    fname = ART_DIR / f"{title_slug or 'article'}_{hsh}.md"
    if fname.exists():  # idempotent: reruns won't duplicate
        return str(fname)
    front = {
        "url": url,
        "title": meta.get("title", ""),
        "published": meta.get("published", ""),
        "source": host,
        "ingested": time.strftime("%Y-%m-%d"),
    }
    blob = f"---\n{json.dumps(front, ensure_ascii=False)}\n---\n\n{text}"
    fname.write_text(blob, encoding="utf-8")
    return str(fname)


def main(
    csv_path: str,
    path_prefix: str,
    limit: int = 50,
    throttle: float = 0.6,
    no_index: bool = False,
) -> None:
    print(f"CSV: {csv_path}")
    df = load_gsc_csv(csv_path)
    urls = pick_top_urls(df, prefix=path_prefix, limit=limit)
    print(f"Selected {len(urls)} URLs from GSC (prefix='{path_prefix}')")

    stats = {"ok": 0, "empty": 0, "http": 0, "too_short": 0}
    for url in urls:
        text, meta, reason = fetch_main_text(url)
        if not text:
            if reason.startswith("http_error"):
                stats["http"] += 1
            else:
                stats["empty"] += 1
            if VERBOSE:
                print("SKIP:", reason, url)
            continue

        if not should_keep(text):
            stats["too_short"] += 1
            if VERBOSE:
                print("SKIP: too_short", len(text), url)
            continue

        out = save_markdown(url, text, meta)
        stats["ok"] += 1
        if VERBOSE:
            print("SAVED:", out)

        if throttle and throttle > 0:
            time.sleep(throttle)

    print("Ingest stats:", stats)

    if not no_index:
        print("Rebuilding index…")
        from src.rag_store import build_index_from_dir

        build_index_from_dir(str(ART_DIR))
        print("Done.")


# ------------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import traceback

    try:
        ap = argparse.ArgumentParser(
            description="Ingest GSC top pages into RAG (saves to data/articles)."
        )
        ap.add_argument("--csv", default=str(DATA_DIR / "gsc_top.csv"))
        ap.add_argument("--prefix", default="/legal-online-sports-betting/")
        ap.add_argument("--limit", type=int, default=50)
        ap.add_argument("--throttle", type=float, default=0.6)
        ap.add_argument(
            "--no-index", action="store_true", help="Skip index rebuild after ingest"
        )

        args = ap.parse_args()
        print("ARGS:", vars(args))
        main(args.csv, args.prefix, args.limit, args.throttle, args.no_index)
    except SystemExit:
        # argparse --help exits cleanly
        pass
    except Exception:
        traceback.print_exc()
        raise
