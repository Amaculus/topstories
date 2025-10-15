# src/internal_links.py
import json
import os
import random
import numpy as np
from typing import List, Dict, Optional
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from .models import InlineLinkSpec

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
STORAGE_DIR = "storage"
INDEX_JSON = os.path.join(STORAGE_DIR, "evergreen_index.json")
INDEX_VEC = os.path.join(STORAGE_DIR, "evergreen_vectors.npy")

# Strategic link categories with priority
LINK_CATEGORIES = {
    "operator_review": 1,    # Highest priority
    "odds": 2,
    "sport_hub": 3,
    "education": 4,
    "general": 5             # Lowest priority
}

# URL patterns for categorization
CATEGORY_PATTERNS = {
    "operator_review": "/online-sports-betting/reviews/",
    "odds": "/odds",
    "sport_hub": ["/nfl", "/nba", "/mlb", "/nhl", "/ncaaf", "/ncaab"],
    "education": "/education/",
}

# Keyword mapping for educational content
EDUCATION_KEYWORDS = {
    "bonus bets": ["bonus-bets", "what-are-bonus-bets"],
    "first bet": ["first-bet-insurance", "first-bet-safety-net"],
    "moneyline": ["moneyline", "what-is-moneyline"],
    "spread": ["point-spread", "spread-betting"],
    "parlay": ["parlay", "what-is-a-parlay"],
    "odds": ["american-odds", "how-to-read-odds"],
    "+ev": ["ev-betting", "expected-value"],
    "arbitrage": ["arbitrage-betting"],
    "bankroll": ["bankroll-management"],
    "responsible": ["responsible-gambling"],
}

def _client():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _embed(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    resp = _client().embeddings.create(model=EMBED_MODEL, input=texts)
    vecs = [d.embedding for d in resp.data]
    return np.asarray(vecs, dtype=np.float32)

def ingest_from_jsonl(path: str):
    """Load evergreen articles from JSONL into the index."""
    os.makedirs(STORAGE_DIR, exist_ok=True)
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            items.append({
                "id": rec["id"],
                "title": rec["title"],
                "url": rec["url"],
                "summary": rec.get("summary", "")
            })
    docs = [f"{r['title']} — {r['summary']}" for r in items]
    vecs = _embed(docs)
    with open(INDEX_JSON, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    np.save(INDEX_VEC, vecs)
    print(f"Indexed {len(items)} evergreen articles")

def _load_index():
    """Load the evergreen index from disk."""
    if not (os.path.exists(INDEX_JSON) and os.path.exists(INDEX_VEC)):
        raise RuntimeError(
            "Evergreen index not found. Run ingest_from_jsonl() first."
        )
    with open(INDEX_JSON, "r", encoding="utf-8") as f:
        items = json.load(f)
    vecs = np.load(INDEX_VEC)
    return items, vecs

def _categorize_link(url: str, brand: str = "") -> str:
    """Categorize a link by its URL pattern."""
    url_lower = url.lower()
    brand_slug = brand.lower().replace(" ", "-") if brand else ""
    
    # Check operator review (highest priority)
    if CATEGORY_PATTERNS["operator_review"] in url_lower:
        if brand_slug and brand_slug in url_lower:
            return "operator_review"
        return "general"  # Review page but not for this operator
    
    # Check odds pages
    if CATEGORY_PATTERNS["odds"] in url_lower:
        return "odds"
    
    # Check sport hubs
    for hub in CATEGORY_PATTERNS["sport_hub"]:
        if url_lower.endswith(hub) or f"{hub}/" in url_lower:
            return "sport_hub"
    
    # Check education
    if CATEGORY_PATTERNS["education"] in url_lower:
        return "education"
    
    return "general"

def _extract_sport_from_context(section_heading: str, facts: List[str]) -> Optional[str]:
    """Extract sport from section context."""
    text = (section_heading + " " + " ".join(facts)).lower()
    
    sport_keywords = {
        "nfl": ["nfl", "football", "chiefs", "packers", "touchdown"],
        "nba": ["nba", "basketball", "lakers", "points"],
        "mlb": ["mlb", "baseball", "yankees", "home run"],
        "nhl": ["nhl", "hockey", "goal"],
        "ncaaf": ["college football", "ncaaf"],
        "ncaab": ["college basketball", "ncaab", "march madness"],
    }
    
    for sport, keywords in sport_keywords.items():
        if any(kw in text for kw in keywords):
            return sport
    
    return None

def _extract_education_keywords(section_heading: str, facts: List[str]) -> List[str]:
    """Extract educational topics from section context."""
    text = (section_heading + " " + " ".join(facts)).lower()
    matches = []
    
    for concept, slugs in EDUCATION_KEYWORDS.items():
        if concept in text:
            matches.extend(slugs)
    
    return matches

def suggest_links_for_brief(
    section_heading: str,
    facts: List[str],
    brand: str = "",
    sport: str = "",
    k: int = 3,
    exclude_urls: set = None,  # NEW - exclude already used links
) -> List[InlineLinkSpec]:
    """
    Strategically suggest k internal links based on content context.
    Prioritizes: operator review > odds > sport hub > education > general
    """
    if exclude_urls is None:
        exclude_urls = set()
    
    try:
        items, vecs = _load_index()
        
        if len(items) == 0:
            print("WARNING: Evergreen index is empty")
            return []
        
        # Detect sport if not provided
        detected_sport = sport or _extract_sport_from_context(section_heading, facts)
        
        # Extract educational keywords
        education_keywords = _extract_education_keywords(section_heading, facts)
        
        # Categorize all available links
        categorized_links = {cat: [] for cat in LINK_CATEGORIES.keys()}
        
        for idx, item in enumerate(items):
            url = item.get("url", "")
            title = item.get("title", "")
            summary = item.get("summary", "")
            
            if not (url and title):
                continue
            
            # Skip if already used
            if url in exclude_urls:
                continue
            
            category = _categorize_link(url, brand)
            
            # Special handling for sport-specific links
            if category == "odds" and detected_sport:
                if f"/{detected_sport}/odds" not in url.lower():
                    category = "general"
            
            if category == "sport_hub" and detected_sport:
                if f"/{detected_sport}" not in url.lower():
                    category = "general"
            
            # Special handling for education keywords
            if category == "education" and education_keywords:
                if not any(kw in url.lower() for kw in education_keywords):
                    category = "general"
            
            # Filter out competitor operator reviews
            if category == "operator_review":
                brand_slug = brand.lower().replace(" ", "-")
                if brand and brand_slug not in url.lower():
                    continue  # Skip other brands' reviews
            
            # Generate anchors
            anchors = [title]
            if len(title.split()) > 5:
                anchors.append(" ".join(title.split()[:5]) + "...")
            if summary:
                words = summary.split()[:8]
                anchor = " ".join(words)
                if anchor not in anchors:
                    anchors.append(anchor)
            
            categorized_links[category].append(InlineLinkSpec(
                title=title,
                url=url,
                recommended_anchors=anchors[:3]
            ))
        
        # Strategic selection: pick from categories by priority
        selected = []
        
        # 1. Always include operator review if available
        if categorized_links["operator_review"]:
            selected.append(categorized_links["operator_review"][0])
        
        # 2. Add odds page if sport detected
        if detected_sport and categorized_links["odds"] and len(selected) < k:
            selected.append(categorized_links["odds"][0])
        
        # 3. Add sport hub if sport detected
        if detected_sport and categorized_links["sport_hub"] and len(selected) < k:
            selected.append(categorized_links["sport_hub"][0])
        
        # 4. Add education link if keywords found
        if education_keywords and categorized_links["education"] and len(selected) < k:
            selected.append(categorized_links["education"][0])
        
        # 5. Fill remaining slots with DIVERSE semantic search
        if len(selected) < k:
            # Build more specific query
            query_parts = [section_heading]
            if brand:
                query_parts.append(brand)
            if detected_sport:
                query_parts.append(detected_sport)
            query_text = " ".join(query_parts).strip()
            
            q_vec = _embed([query_text])[0]
            sims = cosine_similarity([q_vec], vecs)[0]
            
            # Get top semantic matches not already selected
            selected_urls = {link.url for link in selected}
            selected_urls.update(exclude_urls)
            
            top_indices = np.argsort(sims)[::-1]
            
            # Add some randomization to semantic results
            top_20_indices = top_indices[:20]  # Get top 20
            np.random.shuffle(top_20_indices)  # Shuffle them
            
            for idx in top_20_indices:
                if len(selected) >= k:
                    break
                if idx >= len(items):
                    continue
                    
                item = items[idx]
                url = item.get("url", "")
                
                if url in selected_urls:
                    continue
                
                # Skip competitor operator reviews in semantic search too
                if "/reviews/" in url.lower():
                    brand_slug = brand.lower().replace(" ", "-")
                    if brand and brand_slug not in url.lower():
                        continue
                
                title = item.get("title", "")
                summary = item.get("summary", "")
                
                if not (url and title):
                    continue
                
                anchors = [title]
                if len(title.split()) > 5:
                    anchors.append(" ".join(title.split()[:5]) + "...")
                if summary:
                    words = summary.split()[:8]
                    anchor = " ".join(words)
                    if anchor not in anchors:
                        anchors.append(anchor)
                
                selected.append(InlineLinkSpec(
                    title=title,
                    url=url,
                    recommended_anchors=anchors[:3]
                ))
                selected_urls.add(url)
        
        return selected[:k]
    
    except Exception as e:
        print(f"ERROR in suggest_links_for_brief: {e}")
        import traceback
        traceback.print_exc()
        return []