# scripts/rebuild_evergreen_index.py
import json
import os
import hashlib
import numpy as np
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv  # Add this import

# Load environment variables from .env file
load_dotenv()  # Add this line

# Paths
ROOT = Path(__file__).resolve().parents[1]
EVERGREEN_FILE = ROOT / "data" / "evergreen.jsonl"
STORAGE_DIR = ROOT / "storage"
INDEX_JSON = STORAGE_DIR / "evergreen_index.json"
INDEX_VEC = STORAGE_DIR / "evergreen_vectors.npy"

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")
    return OpenAI(api_key=api_key)

def embed_batch(texts, batch_size=100):
    """Embed texts in batches to avoid rate limits"""
    client = get_client()
    all_vecs = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        print(f"Embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        vecs = [d.embedding for d in resp.data]
        all_vecs.extend(vecs)
    
    return np.asarray(all_vecs, dtype=np.float32)

def rebuild_index():
    print(f"Reading evergreen data from: {EVERGREEN_FILE}")
    
    if not EVERGREEN_FILE.exists():
        raise FileNotFoundError(f"Evergreen file not found: {EVERGREEN_FILE}")
    
    # Read all items
    items = []
    with open(EVERGREEN_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                
                # Extract fields from your actual format
                url = rec.get("url", "")
                title = rec.get("title", "")
                description = rec.get("description", "")
                anchors = rec.get("anchors", [])
                
                # Generate an ID from the URL if not present
                if "id" in rec:
                    item_id = rec["id"]
                else:
                    # Create a short hash from the URL
                    item_id = hashlib.md5(url.encode()).hexdigest()[:12]
                
                # Use description as summary, or first anchor if no description
                summary = description or (anchors[0] if anchors else "")
                
                if not title or not url:
                    print(f"Warning: Skipping line {line_num} - missing title or URL")
                    continue
                
                items.append({
                    "id": item_id,
                    "title": title,
                    "url": url,
                    "summary": summary
                })
                
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed JSON on line {line_num}: {e}")
            except Exception as e:
                print(f"Warning: Error on line {line_num}: {e}")
    
    print(f"Loaded {len(items)} items from evergreen.jsonl")
    
    if not items:
        raise ValueError("No valid items found in evergreen.jsonl")
    
    # Create embedding texts
    docs = [f"{r['title']} — {r['summary']}" for r in items]
    
    print("Generating embeddings (this may take a while)...")
    vecs = embed_batch(docs)
    
    # Ensure storage directory exists
    STORAGE_DIR.mkdir(exist_ok=True)
    
    # Save index
    print(f"Saving index to: {INDEX_JSON}")
    with open(INDEX_JSON, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    
    print(f"Saving vectors to: {INDEX_VEC}")
    np.save(INDEX_VEC, vecs)
    
    print(f"✓ Index rebuilt successfully!")
    print(f"  Items: {len(items)}")
    print(f"  Vector shape: {vecs.shape}")

if __name__ == "__main__":
    rebuild_index()