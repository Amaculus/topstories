# src/validators.py
import re
from typing import List, Dict
from src.content_guidelines import get_prohibited_patterns

# Get prohibited patterns from guidelines
BANNED = get_prohibited_patterns()

BET_TRIGGERS = [r"\bbet\b", r"\bwager\b", r"\bparlay\b"]
ALLOWLIST = ["yourdomain.com"]  # REPLACE with your actual internal domain(s)

STATE_DISCLAIMERS = {
    "ALL": "21+. Gambling problem? Call 1-800-GAMBLER. Please bet responsibly.",
    "NY": "21+. Gambling problem? Call 877-8-HOPENY or text HOPENY (467369).",
    "AZ": "21+. Gambling problem? Call 1-800-NEXT-STEP.",
    "PA": "21+. Gambling problem? Call 1-800-GAMBLER.",
    "NJ": "21+. Gambling problem? Call 1-800-GAMBLER.",
    "MI": "21+. Gambling problem? Call 1-800-GAMBLER.",
}

def disclaimer_for_state(state: str) -> str:
    """Get the appropriate responsible gaming disclaimer for a state."""
    return STATE_DISCLAIMERS.get(state.upper(), STATE_DISCLAIMERS["ALL"])

_CTA_LINK = re.compile(r"\[Claim Offer\]\(([^)]+)\)", re.IGNORECASE)

def verify_offer_block(article_md: str, canonical_offer_md: str) -> list[str]:
    """
    Require at least one CTA and ensure at least one matches the canonical URL.
    """
    errs = []
    links = _CTA_LINK.findall(article_md or "")
    if len(links) < 1:
        errs.append("No CTA link found. Add at least one '[Claim Offer](...)' link.")

    canon = _CTA_LINK.findall(canonical_offer_md or "")
    if canon:
        canon_url = canon[0].strip()
        if links and not any(l.strip() == canon_url for l in links):
            errs.append("CTA link URL does not match the canonical offer URL.")
    return errs

def verify_internal_links(md: str, required: List[Dict[str, str]], allowlist: List[str] = ALLOWLIST) -> list[str]:
    """Check internal link usage and compliance."""
    errs = []
    for lk in required:
        url = lk["url"]
        count = md.count(f"]({url})")
        if count != 1:
            errs.append(f"Internal link {url} appears {count} times (must be 1).")
    
    # Check for links in headings
    for h in re.findall(r"^#+ .*", md, flags=re.M):
        if "](" in h:
            errs.append("Link found in a heading.")
    
    # Check external URLs
    for m in re.finditer(r"\]\((https?://[^)]+)\)", md):
        u = m.group(1)
        if not any(dom in u for dom in allowlist):
            errs.append(f"External URL not allowed: {u}")
    
    # Check anchor text length
    for m in re.finditer(r"\[([^\]]+)\]\((https?://[^)]+)\)", md):
        anchor = m.group(1).strip()
        if len(anchor.split()) < 2:
            errs.append(f"Anchor too short: '{anchor}'")
    
    return errs

def compliance_check(md: str) -> list[str]:
    """Check for prohibited phrasing and required disclaimers."""
    errs = []
    
    # Check banned patterns
    for pat in BANNED:
        if re.search(pat, md, flags=re.I):
            errs.append(f"Prohibited phrasing found: pattern /{pat}/")
    
    # Check for responsible gaming mention
    if any(re.search(t, md, flags=re.I) for t in BET_TRIGGERS):
        if "responsible" not in md.lower() and "gambling problem" not in md.lower():
            errs.append("Responsible gaming note missing.")
    
    return errs

def seo_lint(md: str) -> list[str]:
    """Check SEO quality metrics."""
    errs = []
    
    # Check paragraph length
    paras = [p for p in md.split("\n\n") if p.strip()]
    long = [p for p in paras if len(p.split()) > 130]
    if long:
        errs.append(f"{len(long)} paragraph(s) exceed ~120 words.")
    
    # Check link density
    links = len(re.findall(r"\]\((https?://[^)]+)\)", md))
    words = len(md.split())
    if words > 0 and links / words > (1 / 120):
        errs.append("Link density too high (> 1 per ~120 words).")
    
    return errs