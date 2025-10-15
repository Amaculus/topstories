# src/switchboard_links.py
import re
from typing import Optional

def inject_switchboard_links(
    text: str,
    brand: str,
    bonus_code: str,
    switchboard_url: str,
    max_links: int = 12
) -> str:
    """
    Inject clickable switchboard links into text wherever brand + code are mentioned.
    
    Converts: "bet365 bonus code TOPACTION"
    To: "<a ...>bet365 bonus code TOPACTION</a>"
    """
    if not (brand and bonus_code and switchboard_url):
        return text
    
    # Build pattern to match various phrasings
    # Matches: "bet365 bonus code TOPACTION" or "bet365 promo code TOPACTION" etc.
    brand_escaped = re.escape(brand)
    code_escaped = re.escape(bonus_code)
    
    # Pattern matches:
    # - "bet365 bonus code TOPACTION"
    # - "bet365 promo code TOPACTION"  
    # - "the bet365 bonus code TOPACTION"
    # - "bet365 code TOPACTION"
    pattern = re.compile(
        rf'\b(the\s+)?({brand_escaped})\s+(bonus\s+code|promo\s+code|code)\s+({code_escaped})\b',
        re.IGNORECASE
    )
    
    # Build the replacement link
    link_template = (
        r'<a data-id="switchboard_tracking" '
        f'href="{switchboard_url}" '
        r'rel="nofollow">'
        r'<strong>\2 \3 \4</strong>'  # \2=brand, \3=code type, \4=code
        r'</a>'
    )
    
    # Track how many links we've injected
    links_injected = 0
    
    def replacer(match):
        nonlocal links_injected
        if links_injected >= max_links:
            return match.group(0)  # Don't replace, keep original
        links_injected += 1
        
        # Build link with proper groups
        the_prefix = match.group(1) or ""
        brand_text = match.group(2)
        code_type = match.group(3)
        code_text = match.group(4)
        
        return (
            f'{the_prefix}'
            f'<a data-id="switchboard_tracking" '
            f'href="{switchboard_url}" '
            f'rel="nofollow">'
            f'<strong>{brand_text} {code_type} {code_text}</strong>'
            f'</a>'
        )
    
    result = pattern.sub(replacer, text)
    return result


def inject_brand_links(
    text: str,
    brand: str,
    review_url: Optional[str] = None,
    max_links: int = 3
) -> str:
    """
    Inject links to brand review page when brand name mentioned alone.
    
    Converts: "bet365" -> "<a href='/reviews/bet365'>bet365</a>"
    Only when NOT already part of "bet365 bonus code"
    """
    if not (brand and review_url):
        return text
    
    brand_escaped = re.escape(brand)
    
    # Match brand name NOT followed by "bonus code" or "promo code"
    # Negative lookahead to avoid double-linking
    pattern = re.compile(
        rf'\b({brand_escaped})(?!\s+(?:bonus|promo|code)\s+code)',
        re.IGNORECASE
    )
    
    links_injected = 0
    
    def replacer(match):
        nonlocal links_injected
        
        # Skip if already inside an <a> tag
        before_text = text[:match.start()]
        if '<a' in before_text and '</a>' not in before_text[before_text.rfind('<a'):]:
            return match.group(0)
        
        if links_injected >= max_links:
            return match.group(0)
        
        links_injected += 1
        return f'<a href="{review_url}" rel="follow">{match.group(1)}</a>'
    
    result = pattern.sub(replacer, text)
    return result