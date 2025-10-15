# app.py – Single textbox flow: Offer → Title → Outline(text) → Draft(in-place) + Inline Export

import os
import re
import io
import json
import uuid
import requests
import streamlit as st
from bs4 import BeautifulSoup
from datetime import datetime
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Env & defaults
# -----------------------------------------------------------------------------
load_dotenv()
if not os.getenv("OFFERS_SHEET_TAB") and not os.getenv("OFFERS_WORKSHEET"):
    os.environ["OFFERS_SHEET_TAB"] = "Ultimate Builder"

# -----------------------------------------------------------------------------
# Local modules
# -----------------------------------------------------------------------------
from src.rag_store import query_articles
from src.offers_layer import get_offers_df_cached, get_offer_by_id, render_offer_block
from src.internal_links import suggest_links_for_brief
from src.validators import disclaimer_for_state
from src.prompt_factory import make_promptsect, make_intro_prompt
from src.models import SectionBrief
from src.llm import generate_markdown
from src.content_guidelines import get_temperature_by_section

# ---------------- Page config ----------------
st.set_page_config(page_title="Plan-Then-Write", layout="wide")
st.title("Plan-Then-Write – Outline → Draft (single box)")

# ---------------- Session state ----------------
def init_state():
    ss = st.session_state
    ss.setdefault("offer_sel", {})
    ss.setdefault("user_title", "")
    ss.setdefault("keyword", "")
    ss.setdefault("box_mode", "outline")
    ss.setdefault("article_box", "")
    ss.setdefault("tokens_cached", [])
init_state()
ss = st.session_state

# ---------------- Helpers ----------------
def today_long(tz: str = "US/Eastern") -> str:
    """Return 'Thursday, September 25, 2025' in the given timezone."""
    try:
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo(tz))
    except Exception:
        now = datetime.now()
    return f"{now.strftime('%A')}, {now.strftime('%B')} {now.day}, {now.year}"

def scrape_competitor(url: str) -> str:
    """Fetch headings/paras for prompt context."""
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (PTW)"}, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        parts = []
        for tag in soup.select("h1, h2, h3, p, li"):
            t = tag.get_text(" ", strip=True)
            if t and len(t) > 1:
                parts.append(t)
        return "\n".join(parts[:800])
    except Exception as e:
        return f"[FETCH_FAILED] {url} :: {e}"

# ---- Outline token utilities ------------------------------------------------------
TOKEN_RE = re.compile(r"^\s*\[(?P<label>intro|shortcode|h[1-6])(?:\s*:\s*(?P<title>.+?))?\]\s*$", re.I)

def parse_outline_tokens(text: str) -> list[dict]:
    """Parse plain-text outline into tokens."""
    tokens = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = TOKEN_RE.match(line)
        if not m:
            continue
        label = m.group("label").lower()
        title = (m.group("title") or "").strip()
        if label == "intro":
            level = "intro"
        elif label == "shortcode":
            level = "shortcode"
        else:
            level = label
        tokens.append({"level": level, "title": title})
    return tokens

def tokens_to_outline_text(tokens: list[dict]) -> str:
    """Render tokens back to outline notation."""
    out = []
    for t in tokens:
        lvl = t["level"]
        if lvl == "intro":
            out.append("[INTRO]")
        elif lvl == "shortcode":
            out.append("[SHORTCODE]")
        else:
            out.append(f"[{lvl.upper()}: {t['title']}]")
    return "\n".join(out)

# ---- AI Outline (produces tokens for the box) ------------------------------------
def ai_outline_tokens_rag(
    *,
    keyword: str,
    exact_title: str | None = None,
    brand: str | None = None,
    offer_text: str | None = None,
    page_type_hint: str | None = None,
    comp_urls_txt: str | None = None,
) -> list[dict]:
    """Build an outline using RAG and competitor context."""
    # Gather RAG snippets
    try:
        hits = query_articles(keyword, k=6, snippet_chars=800)
    except Exception:
        hits = []
    rag_snips = [(h.get("snippet") or "").strip() for h in hits if isinstance(h, dict)]
    rag_ctx = "\n\n".join([s for s in rag_snips if s])[:4000] if rag_snips else ""

    working_title = (exact_title or "").strip()
    sys = (
        "You are an SEO content planner for sports betting promo articles. "
        "Your job is to propose a clear outline, not prose."
    )

    user = f"""
KEYWORD / FOCUS:
{keyword}

WORKING TITLE (use as H1):
{working_title or "[generate a sensible H1 from the keyword]"}

PAGE TYPE HINT:
{(page_type_hint or "promo").strip()}

BRAND + OFFER (for context; do NOT write a CTA here):
{(brand or "").strip()} – {(offer_text or "").strip()}

COMPETITOR CONTEXT (structure/topics only; IGNORE their writing style):
{comp_urls_txt or "[none]"}

OUR HOUSE STYLE (match THIS voice when writing):
{rag_ctx or "[none]"}

OUTPUT FORMAT (STRICT):
- One item per line.
- Use bracketed tokens:
  [INTRO]
  [SHORTCODE]     <-- render promo module here
  [H2: ...]
  [H3: ...]  (only when helpful under the preceding H2)
- Start with a single [INTRO]; keep intro 2–3 sentences.
- Insert [SHORTCODE] tokens where promo cards should appear (one after intro, then 2-3 throughout)
- Then propose as many H2 as needed. Avoid duplicate/near-duplicate headings.
- Do NOT include CTAs in the outline (those are inserted during drafting).
- Keep headings short, concrete, and scannable.
"""

    outline_text = generate_markdown(
        system=sys, 
        user=user, 
        temperature=get_temperature_by_section("outline")
    )
    tokens = parse_outline_tokens(outline_text)

    # Fallback if parsing failed
    if not tokens:
        tokens = [
            {"level": "intro", "title": ""},
            {"level": "shortcode", "title": ""},
            {"level": "h2", "title": "Overview"},
            {"level": "shortcode", "title": ""},
            {"level": "h2", "title": "How to Claim the Offer"},
            {"level": "shortcode", "title": ""},
            {"level": "h2", "title": "Eligibility & Where It's Live"},
            {"level": "h2", "title": "Offer Terms & Key Details"},
            {"level": "shortcode", "title": ""},
            {"level": "h2", "title": "How to Sign Up"},  # NEW - will be made dynamic
        ]
    return tokens

# ---- Draft generation from tokens -------------------------------------------------
    
def _build_terms_section(offer_row: dict | None, state: str) -> str:
    """Build the final terms & conditions + responsible gaming section."""
    if not offer_row:
        return ""
    
    brand = (offer_row.get("brand") or "").strip()
    terms = (offer_row.get("terms") or "").strip()
    
    # Start with heading
    parts = ["## Terms and Conditions"]
    
    # Add offer-specific terms if available
    if terms:
        # Clean up the terms text
        terms_clean = terms.replace("\\n", "\n").strip()
        parts.append(f"{terms_clean}")
    else:
        # Generic fallback if no terms provided
        parts.append(f"Please review the full terms and conditions on the {brand} website before signing up.")
    
    # Add responsible gaming (always included)
    parts.append("## Responsible Gaming")
    parts.append(
        "Always bet responsibly. If you or someone you know has a gambling problem, "
        "help is available. Call 1-800-GAMBLER or visit the National Council on Problem Gambling."
    )
    
    # Add age requirement
    parts.append("This offer is available to users 21+ only. Terms and conditions apply.")
    
    return "\n\n".join(parts)


def generate_article_from_tokens(
    tokens: list[dict],
    title: str,
    offer_row: dict | None,
    state: str,
    event_context: str = "",  # NEW
    sport: str = "",  # NEW
    switchboard_url: str = "",
    target_date: datetime = None,
) -> str:
    """Generate full Markdown from tokens."""
    parts = []
    previous_content = ""
    used_link_urls = set()
    
    # Extract brand and detect sport for strategic link selection
    # Extract brand (sport is passed as parameter)
    brand = ""
    if offer_row:
        brand = (offer_row.get("brand") or "").strip()
    
    if title:
        parts.append(f"# {title}")

    # Get states list for passing to prompts
    available_states = (offer_row or {}).get("states_list", [])

    # Intro (lede) with auto-date
    if any(t["level"] == "intro" for t in tokens):
        # Use target date if provided, otherwise today
        if target_date:
            date_str = f"{target_date.strftime('%A')}, {target_date.strftime('%B')} {target_date.day}, {target_date.year}"
        else:
            date_str = today_long("US/Eastern")
        ps_intro = make_intro_prompt(
            brand=(offer_row.get("brand") if offer_row else "") or "",
            offer_text=(offer_row.get("offer_text") if offer_row else "") or "",
            bonus_code=(offer_row.get("bonus_code") if offer_row else "") or "",
            date_str=date_str,
            available_states=available_states,
            event_context=event_context,

        )
        intro_md = generate_markdown(
            ps_intro.system, 
            ps_intro.user, 
            temperature=ps_intro.temperature
        ).strip()
        if intro_md.startswith("#"):
            intro_md = intro_md.lstrip("# ").strip()
        parts.append(intro_md)
        previous_content = intro_md

    # Sections
    section_count = 0
    for t in tokens:
        if t["level"] == "intro":
            continue
            
        if t["level"].lower() == "shortcode":
            shortcode = (offer_row or {}).get("shortcode", "") if offer_row else ""
            shortcode = (shortcode or "").strip()
            if shortcode:
                parts.append(shortcode)
            else:
                parts.append(render_offer_block(offer_row or {}))
            continue
            
        heading = t["title"] or ""
        heading_lower = heading.lower()
        if "sign up" in heading_lower and "how to" in heading_lower:
            # Build dynamic heading
            keyword = ss.get("keyword", "").strip() or ss.get("user_title", "").strip()
            date_str = today_long("US/Eastern")
            
            if keyword:
                heading = f"How to Sign Up for the {keyword} on {date_str}"
            else:
                heading = f"How to Sign Up on {date_str}"

        
        section_count += 1
        
        # Section-specific objectives to prevent repetition
        heading_lower = heading.lower()
        if "overview" in heading_lower:
            objective = (
                f"Write about WHY this offer is valuable to bettors. "
                f"Focus on what makes it appealing or how it compares to similar offers. "
                f"DO NOT restate the basic offer details (already covered in intro). "
                f"Keep to 3-4 sentences max."
            )

        elif "sign up" in heading_lower and "how to" in heading_lower:
            # NEW - Numbered step-by-step registration guide
            brand = (offer_row or {}).get("brand", "").strip()
            bonus_code = (offer_row or {}).get("bonus_code", "").strip()
            objective = (
                f"CRITICAL: Output ONLY a numbered list. NO PARAGRAPHS.\n\n"
                f"Provide exactly 5 numbered steps (1. 2. 3. 4. 5.) for signing up with {brand}.\n"
            )
        elif "claim" in heading_lower or "how to" in heading_lower:
            objective = (
                f"Provide a WORKED EXAMPLE using the featured game: {event_context}. "
                f"Use first-person: 'For example, if I place a $50 bet on [specific event] at [odds]...' "
                f"Show both win and loss scenarios with exact profit calculations. "
                f"DO NOT restate the basic offer - the intro already covered that. Focus on the EXAMPLE."
            )
        elif "steps" in heading_lower:
            objective = (
                f"Provide a numbered step-by-step list (1, 2, 3...) for claiming the offer. "
                f"Be procedural and concise. Each step should be one sentence."
            )
        elif "eligibility" in heading_lower or "where" in heading_lower or "live" in heading_lower:
            objective = (
                f"Briefly state who qualifies (new users, age requirement). "
                f"The states list was already mentioned in the intro, so just confirm eligibility. "
                f"Keep to 2-3 sentences max. Focus on restrictions or special cases."
            )
        elif "terms" in heading_lower or "details" in heading_lower or "key" in heading_lower:
            objective = (
                f"Cover the fine print: minimum odds requirements, bonus expiration, wagering rules. "
                f"Focus on RESTRICTIONS and LIMITATIONS only. "
                f"Skip restating the offer amount or states - focus on what users need to know about terms. "
                f"3-4 sentences max."
            )
        elif "responsible" in heading_lower or "gaming" in heading_lower:
            objective = (
                f"Brief reminder about responsible gambling with the helpline. "
                f"Keep to 2-3 sentences max. No need to restate the offer."
            )


        else:
            objective = f"Write the section under the heading EXACTLY titled '{heading}'."
        
        brief = SectionBrief(
            section_id=f"sec-{uuid.uuid4().hex[:6]}",
            objective=objective,
            audience="Beginner–intermediate US sports bettors (ages 21-65)",
            constraints={},
            facts_and_points=[f"Featured game: {event_context}"] if event_context and ("claim" in heading_lower or "how to" in heading_lower) else [],
            retrieved_snippets=[],
        )
        
        # RAG retrieval - reduce influence for later sections
        query_parts = [
            heading,
            "sports betting promo" if "claim" in heading_lower else "sportsbook",
            (offer_row or {}).get('brand', ''),
        ]
        query = " ".join(p for p in query_parts if p.strip())
        
        try:
            # Less RAG influence for later sections to reduce repetition
            if section_count > 2:
                _hits = query_articles(query, k=3, snippet_chars=300)
            else:
                _hits = query_articles(query, k=8, snippet_chars=400)
            
            # Filter by relevance score
            relevant = [h for h in _hits if h.get('score', 0) > 0.35][:5]
            snips = [(h.get("snippet") or "").strip() for h in relevant]
            brief.retrieved_snippets = [{"snippet": s} for s in snips if s]
        except Exception as e:
            if os.getenv("DEBUG"):
                st.warning(f"RAG retrieval failed for '{heading}': {e}")


        # In generate_article_from_tokens, around line 277, BEFORE the suggest_links_for_brief call:

        if os.getenv("DEBUG"):
            print("---")
            print(f"**DEBUG: Section '{heading}'**")
            print(f"- Brand: {brand}")
            print(f"- Sport: {sport}")

        # Strategic link selection with brand and sport context
        inline_links = suggest_links_for_brief(
            section_heading=heading,
            facts=[],
            brand=brand,
            sport=sport,
            k=3,
            exclude_urls=used_link_urls 
        )

        for link in inline_links:
            used_link_urls.add(link.url)

        if os.getenv("DEBUG"):
            print(f"- Returned {len(inline_links)} links:")
            for link in inline_links:
                print(f"  - {link.title} → {link.url}")
            print("---")
        
        disclaimer = disclaimer_for_state(state)

        ps = make_promptsect(
            brief, 
            offer_row or {}, 
            inline_links, 
            disclaimer,
            previous_content=previous_content[-1500:],
            available_states=available_states,
        )
        
        body_md = generate_markdown(
            ps.system, 
            ps.user, 
            temperature=ps.temperature
        ).strip()
        
        # Post-generation checks for repetition
        if previous_content and os.getenv("DEBUG"):
            # Check if repeating states list unnecessarily
            if available_states and len(available_states) > 5:
                states_mentioned = sum(1 for s in available_states if s in body_md)
                if states_mentioned > 5:
                    st.warning(f"⚠️ Section '{heading}' may be repeating the full states list")
            
            # Check if repeating promo code too much
            bonus_code = (offer_row or {}).get("bonus_code", "")
            if bonus_code and body_md.count(bonus_code) > 1:
                st.warning(f"⚠️ Section '{heading}' mentions promo code {bonus_code} multiple times")
        
        level = t["level"].lower()
        prefix = "##" if level == "h2" else "###" if level == "h3" else "####"
        section_text = f"{prefix} {heading}\n\n{body_md}"
        parts.append(section_text)
        previous_content += f"\n\n{section_text}"

    full_article = "\n\n".join(parts).strip()

            # Inject switchboard links if available
    if switchboard_url and offer_row:
            from src.switchboard_links import inject_switchboard_links
            
            brand = (offer_row.get("brand") or "").strip()
            bonus_code = (offer_row.get("bonus_code") or "").strip()
            
            if brand and bonus_code:
                full_article = inject_switchboard_links(
                    text=full_article,
                    brand=brand,
                    bonus_code=bonus_code,
                    switchboard_url=switchboard_url,
                    max_links=12  # Limit to 12 clickable links
                )
                
                st.success(f"✅ Injected switchboard links for '{brand} bonus code {bonus_code}'")


            # Add terms and conditions section at the end
    terms_section = _build_terms_section(offer_row, state)
    if terms_section:
        full_article = f"{full_article}\n\n{terms_section}"
        
    return full_article




# ---- Offer -> Title -> Outline/Draft ----
st.header("1) Offer → Title → Outline/Draft")

cols = st.columns([7, 1])
with cols[0]:
    offers_df = get_offers_df_cached()
with cols[1]:
    if st.button("↻", help="Refresh offers from Google Sheet"):
        # Clear Streamlit cache
        try:
            get_offers_df_cached.clear()
        except Exception:
            pass
        
        # Clear file-based cache
        cache_file = "data/offers_cache.pkl"
        if os.path.exists(cache_file):
            os.remove(cache_file)
            st.success("Cache cleared! Reloading from Google Sheets...")
        
        st.rerun()

if offers_df.empty:
    st.error("No offers loaded. Check OFFERS_SHEET_ID / OFFERS_SHEET_TAB and service-account access.")
    st.stop()

# Build dropdown
# Build dropdown - FILTER OUT CASINO OFFERS
opt_pairs = []
for _, r in offers_df.iterrows():
    # Skip casino offers
    affiliate_offer = str(r.get("affiliate_offer", "")).strip().lower()
    offer_text = str(r.get("offer_text", "")).strip().lower()
    page_type = str(r.get("page_type", "")).strip().lower()
    
    # Filter criteria - skip if contains "casino" in key fields
    if any(keyword in text for text in [affiliate_offer, offer_text, page_type] 
           for keyword in ["casino", "slots", "live dealer"]):
        continue
    
    # Build label for dropdown
    affiliate_offer = str(r.get("affiliate_offer", "")).strip()  # Get original case
    brand_txt = str(r.get("brand", "")).strip()
    code_txt = str(r.get("bonus_code", "")).strip()
    offer_text_display = str(r.get("offer_text", "")).strip()

    main = affiliate_offer or offer_text_display or "(Untitled offer)"
    bits = [main]
    if brand_txt:
        bits.append(f"[{brand_txt}]")
    if code_txt:
        bits.append(f"(code: {code_txt})")

    label = " ".join(bits)
    opt_pairs.append((str(r["offer_id"]), label))


sel_idx = st.selectbox(
    "Offer",
    options=list(range(len(opt_pairs))),
    format_func=lambda i: opt_pairs[i][1],
    key="offer_select_idx",
)
sel_offer_id = opt_pairs[sel_idx][0]
offer_row = get_offer_by_id(offers_df, sel_offer_id)

# Extract switchboard URL from offer sheet
switchboard_url = (offer_row.get("switchboard_link") or "").strip()

if not switchboard_url:
    # Fallback if missing
    st.warning("No switchboard link found for this offer")
    switchboard_url = "https://switchboard.actionnetwork.com/offers?affiliateId=174&context=web-article-top-stories"

# Store for later use
ss["switchboard_url"] = switchboard_url

# Show it for debugging
st.caption(f"**Switchboard URL:** {switchboard_url[:80]}...")

# Display available states (no picker needed)
states_for_offer = offer_row.get("states_list", []) or []
if states_for_offer and states_for_offer != ["ALL"]:
    st.caption(f"**Available states for this offer:** {', '.join(states_for_offer)}")
else:
    st.caption("**Available states for this offer:** Nationwide")

# No state selection - always use generic disclaimer
sel_state = "ALL"

# Sport selector for featured game
st.subheader("Sport Selection")
sport_options = {
    "NFL": "nfl",
    "NBA": "nba",
    "MLB": "mlb",
    "NHL": "nhl",
}

col_sport, col_date = st.columns([1, 1])

with col_sport:
    selected_sport_label = st.selectbox(
        "Sport",
        options=list(sport_options.keys()),
        index=0,
        help="Select the sport for fetching games"
    )

sport_selected = sport_options[selected_sport_label]

with col_date:
    from datetime import date, timedelta
    
    # Date picker (default to today, allow up to 7 days ahead)
    target_date = st.date_input(
        "Game Date",
        value=date.today(),
        min_value=date.today(),
        max_value=date.today() + timedelta(days=7),
        help="Select the date to fetch games and write content for"
    )

# Convert date to datetime for API
from datetime import datetime
from zoneinfo import ZoneInfo

target_datetime = datetime.combine(target_date, datetime.min.time())
target_datetime = target_datetime.replace(tzinfo=ZoneInfo("America/New_York"))

# Fetch games for selected date and sport
from src.event_fetcher import (
    get_games_for_date, 
    filter_prime_time_games, 
    format_event_for_prompt,
    format_game_for_dropdown
)

event_context = ""
selected_game = None

with st.spinner(f"Fetching {selected_sport_label} games for {target_date.strftime('%A, %B %d')}..."):
    try:
        # Fetch ALL games for the date
        all_games = get_games_for_date(sport_selected, target_datetime)
        
        if not all_games:
            st.info(f"No {selected_sport_label} games found for {target_date.strftime('%A, %B %d')} - generating evergreen content")
        else:
            # Identify prime time games (but still show all in dropdown)
            prime_games = filter_prime_time_games(all_games)
            
            # Determine default selection (prime time if available, otherwise first game)
            if prime_games:
                default_game = prime_games[0]
                st.success(f"🏈 Found {len(all_games)} total game(s) - Auto-selected prime time")
            else:
                default_game = all_games[0]
                st.info(f"📋 Found {len(all_games)} game(s) - No prime time games available")
            
            # Find index of default game in all_games list
            try:
                default_idx = all_games.index(default_game)
            except ValueError:
                default_idx = 0
            
            # Build dropdown with ALL games
            game_options = [format_game_for_dropdown(g) for g in all_games]
            
            # Show dropdown with all games, default to prime time
            selected_game_idx = st.selectbox(
                f"Select Game from {len(all_games)} available",
                options=list(range(len(all_games))),
                format_func=lambda i: game_options[i],
                index=default_idx,
                help="All games shown - prime time auto-selected if available"
            )
            
            # Get the user's selected game
            selected_game = all_games[selected_game_idx]
            event_context = format_event_for_prompt(selected_game, target_datetime)
            
            # Show what was selected
            is_prime = selected_game in prime_games if prime_games else False
            prime_badge = "⭐ PRIME TIME" if is_prime else ""
            st.info(f"📅 **Selected:** {event_context} {prime_badge}")
    
    except Exception as e:
        st.warning(f"Failed to fetch games: {e}")
        event_context = ""



ss["user_title"] = st.text_input("Title (H1)", value=ss.get("user_title",""))
ss["keyword"] = st.text_input("Keyword / Focus", value=ss.get("keyword",""))

# Outline helpers
archetype = st.selectbox("Archetype", ["Single-Promo Event", "Roundup", "Legal/Industry Update"], index=0)
st.caption("Optional: paste competitor URLs (one per line). We'll extract headings/paragraphs for context.")
comp_urls_txt = st.text_area("Competitor URLs (optional)", height=80)

# Generate Outline with AI
if st.button("Generate Outline with AI"):
    comp_parts = []
    for url in comp_urls_txt.splitlines():
        if url.strip():
            scraped = scrape_competitor(url.strip())
            comp_parts.append(f"URL: {url}\n{scraped[:1500]}")
    comp_text = "\n\n".join(comp_parts)[:6000]
    
    tokens = ai_outline_tokens_rag(
        keyword=ss["keyword"].strip() or ss["user_title"].strip(),
        exact_title=ss["user_title"].strip(),
        brand=offer_row.get("brand", ""),
        offer_text=offer_row.get("offer_text", ""),
        page_type_hint=(offer_row.get("page_type_clean") or offer_row.get("page_type") or "").strip(),
        comp_urls_txt=comp_text,
    )
    ss["tokens_cached"] = tokens
    ss["box_mode"] = "outline"
    ss["pending_article_box"] = tokens_to_outline_text(tokens)
    st.success("Outline generated. Edit freely, then click Generate Draft.")
    st.rerun()

# Hydrate textbox from pending buffer
ss.setdefault("article_box", "")
if "pending_article_box" in ss:
    ss["article_box"] = ss.pop("pending_article_box")

hint = (
    "Format headings as `[HX: Text]` – e.g., `[INTRO]`, `[H2: How to Claim]`, `[H3: Eligibility]`.\n"
    "Lines that don't match this pattern are ignored during generation."
    if ss["box_mode"] == "outline"
    else "This is the full draft. Edit freely, or click Regenerate Draft to rewrite from the last outline."
)
st.text_area("Outline / Draft", key="article_box", height=360, help=hint)

# ===================== Inline row: Generate/Regenerate + Export ====================
gen_label = "Generate Draft" if ss["box_mode"] == "outline" else "Regenerate Draft"
colGen, colFmt, colName, colDL = st.columns([1, 1.3, 2.2, 1])

# Generate / Regenerate button
if colGen.button(gen_label, type="primary"):
    if ss["box_mode"] == "outline":
        tokens = parse_outline_tokens(ss["article_box"])
        if not tokens:
            st.warning("No valid outline tokens found. Add lines like [INTRO], [H2: How to Claim].")
        else:
            ss["tokens_cached"] = tokens
    tokens = ss.get("tokens_cached", [])
    if tokens:
        full_md = generate_article_from_tokens(
            tokens=tokens,
            title=ss["user_title"].strip(),
            offer_row=offer_row,
            state=sel_state,
            event_context=event_context,  # NEW - pass event context
            sport=sport_selected,  # NEW - pass selected sport
            switchboard_url=ss.get("switchboard_url", ""),
            target_date=target_datetime,
        )
        ss["box_mode"] = "draft"
        ss["pending_article_box"] = full_md
        st.success("Draft generated.")
        st.rerun()
    else:
        st.error("No outline available to generate from.")

# Export controls on the same row
export_fmt = colFmt.selectbox("Format", ["Markdown (.md)", "HTML (.html)", "Word (.docx)"], index=0, key="fmt_inline")
export_name = colName.text_input("File name (no extension)", "draft_article", key="fname_inline")
content_now = ss.get("article_box", "").strip()

# Prepare bytes only when needed
if export_fmt.startswith("Markdown"):
    data_bytes = content_now.encode("utf-8")
    file_ext = ".md"
elif export_fmt.startswith("HTML"):
    html = f"<article>\n{content_now.replace('\n','<br>')}\n</article>"
    data_bytes = html.encode("utf-8")
    file_ext = ".html"
else:
    from docx import Document
    buf = io.BytesIO()
    doc = Document()
    for para in (content_now or "").split("\n\n"):
        doc.add_paragraph(para)
    doc.save(buf)
    data_bytes = buf.getvalue()
    file_ext = ".docx"

colDL.download_button(
    "Download",
    data=data_bytes,
    file_name=f"{export_name or 'draft_article'}{file_ext}",
    disabled=(not bool(content_now)),
    key="dl_inline",
)

st.divider()
st.caption("Tip: Keep editing in the box above; re-generate to refresh from your outline tokens. Export is always available here.")