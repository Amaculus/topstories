# app.py — Compact, minimal UI

import os
import re
import io
import json
import uuid
import requests
import streamlit as st
from bs4 import BeautifulSoup
from datetime import datetime, date, timedelta
from dotenv import load_dotenv
from zoneinfo import ZoneInfo
import markdown

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
from src.event_fetcher import (
    get_games_for_date, 
    filter_prime_time_games, 
    format_event_for_prompt,
    format_game_for_dropdown
)

# ---------------- Page config ----------------
st.set_page_config(page_title="Plan-Then-Write", layout="wide")

# ---------------- Minimal CSS ----------------
st.markdown("""
<style>
    /* Compact spacing */
    .stSelectbox, .stTextInput, .stDateInput {
        margin-bottom: 0.5rem;
    }
    .stSelectbox label, .stTextInput label, .stDateInput label {
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 2px;
    }
    /* Reduce container padding */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }
    /* Tighter columns */
    [data-testid="stHorizontalBlock"] {
        gap: 0.75rem;
    }
    /* Compact text area */
    .stTextArea textarea {
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📝 Content Generator")

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
    try:
        now = datetime.now(ZoneInfo(tz))
    except Exception:
        now = datetime.now()
    return f"{now.strftime('%A')}, {now.strftime('%B')} {now.day}, {now.year}"

def scrape_competitor(url: str) -> str:
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        parts = [tag.get_text(" ", strip=True) for tag in soup.select("h1, h2, h3, p, li") if tag.get_text(strip=True)]
        return "\n".join(parts[:800])
    except Exception as e:
        return f"[FETCH_FAILED] {url} :: {e}"

TOKEN_RE = re.compile(r"^\s*\[(?P<label>intro|shortcode|h[1-6])(?:\s*:\s*(?P<title>.+?))?\]\s*$", re.I)

def parse_outline_tokens(text: str) -> list[dict]:
    tokens = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        m = TOKEN_RE.match(line)
        if not m:
            continue
        label = m.group("label").lower()
        title = (m.group("title") or "").strip()
        level = "intro" if label == "intro" else "shortcode" if label == "shortcode" else label
        tokens.append({"level": level, "title": title})
    return tokens

def tokens_to_outline_text(tokens: list[dict]) -> str:
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

def ai_outline_tokens_rag(*, keyword: str, exact_title: str | None = None, brand: str | None = None, 
                          offer_text: str | None = None, page_type_hint: str | None = None, 
                          comp_urls_txt: str | None = None) -> list[dict]:
    try:
        hits = query_articles(keyword, k=6, snippet_chars=800)
    except Exception:
        hits = []
    rag_snips = [(h.get("snippet") or "").strip() for h in hits if isinstance(h, dict)]
    rag_ctx = "\n\n".join([s for s in rag_snips if s])[:4000] if rag_snips else ""

    working_title = (exact_title or "").strip()
    sys = "You are an SEO content planner for sports betting promo articles. Your job is to propose a CONCISE outline, not prose."

    user = f"""
KEYWORD / FOCUS:
{keyword}

WORKING TITLE (use as H1):
{working_title or "[generate from keyword]"}

PAGE TYPE: {(page_type_hint or "promo").strip()}
BRAND + OFFER: {(brand or "").strip()} — {(offer_text or "").strip()}
COMPETITOR CONTEXT: {comp_urls_txt or "[none]"}
OUR HOUSE STYLE: {rag_ctx or "[none]"}

OUTPUT FORMAT (STRICT):
- One item per line.
- Use bracketed tokens:
  [INTRO]
  [SHORTCODE]     <-- render promo module here
  [H2: ...]
  [H3: ...]  (only when helpful under the preceding H2)
- Start with a single [INTRO]; keep intro 2–3 sentences.
- Insert [SHORTCODE] tokens where promo cards should appear (one after intro, then 3-4 throughout)
- Then propose as many H2 as needed. Avoid duplicate/near-duplicate headings.
- Do NOT include CTAs in the outline (those are inserted during drafting).
- Keep headings short, concrete, and scannable.

CRITICAL CONSTRAINTS:
- Must include a How To Sign Up section at the end
- NEVER INCLUDE OFFER TERMS HEADING
- Do not Include a Responsable Gaming section
- Maximum 2 H3 subsections under any one H2
- Focus on the ESSENTIAL information only
- This is a promo article, not a comprehensive guide
"""

    outline_text = generate_markdown(system=sys, user=user, temperature=get_temperature_by_section("outline"))
    tokens = parse_outline_tokens(outline_text)

    # Post-process to enforce limits
    if tokens:
        filtered = []
        h2_count = 0
        current_h2_h3s = 0
        
        for t in tokens:
            if t["level"] == "intro" or t["level"] == "shortcode":
                filtered.append(t)
            elif t["level"] == "h2":
                h2_count += 1
                current_h2_h3s = 0
                if h2_count <= 6:  # Max 6 H2s
                    filtered.append(t)
            elif t["level"] == "h3":
                current_h2_h3s += 1
                if current_h2_h3s <= 2:  # Max 2 H3s per H2
                    filtered.append(t)
        
        tokens = filtered

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
            {"level": "h2", "title": "How to Sign Up"}, 
        ]
    return tokens

def _build_terms_section(offer_row: dict | None, state: str) -> str:
    if not offer_row:
        return ""
    brand = (offer_row.get("brand") or "").strip()
    terms = (offer_row.get("terms") or "").strip()
    parts = ["## Terms and Conditions"]
    if terms:
        parts.append(terms.replace("\\n", "\n").strip())
    else:
        parts.append(f"Review full terms on the {brand} website.")
    return "\n\n".join(parts)

def _build_default_title(offer_row: dict, sport: str = "", event_context: str = "") -> str:
    brand = (offer_row.get("brand") or "").strip()
    title = f"{brand} Promo Code" if brand else "Sportsbook Promo"
    if event_context:
        matchup = re.search(r'(.+?)\s+on\s+', event_context)
        if matchup:
            title = f"{title}: {matchup.group(1).strip()}"
    elif sport:
        sport_names = {"nfl": "NFL", "nba": "NBA", "mlb": "MLB", "nhl": "NHL"}
        title = f"{title} for {sport_names.get(sport.lower(), sport.upper())}"
    return title

def generate_article_from_tokens(tokens: list[dict], title: str, offer_row: dict | None, state: str,
                                  event_context: str = "", sport: str = "", switchboard_url: str = "",
                                  target_date: datetime = None) -> str:
    parts = []
    previous_content = ""
    brand = (offer_row.get("brand") or "").strip() if offer_row else ""
    
    if title:
        parts.append(f"# {title}")

    available_states = (offer_row or {}).get("states_list", [])

    # CHANGE 1: Single pass for all internal links BEFORE generating sections
    # Collect all section headings and context
    all_section_contexts = []
    for t in tokens:
        if t["level"] not in ["intro", "shortcode"]:
            heading = t["title"] or ""
            all_section_contexts.append(heading)
    
    # Generate ALL internal links at once
    combined_query = " ".join(all_section_contexts[:5])  # Use first 5 headings for context
    if not combined_query.strip():
        combined_query = title or ss.get("keyword", "sports betting promo")
    
    st.info("🔗 Generating internal links for entire article...")
    try:
        # Get 2x the number of sections to ensure we have enough links
        num_links_needed = min(len(all_section_contexts) * 2, 15)
        all_internal_links = suggest_links_for_brief(
            section_heading=combined_query,
            facts=[event_context] if event_context else [],
            brand=brand,
            sport=sport,
            k=num_links_needed,
            exclude_urls=set()
        )
        st.success(f"✅ Generated {len(all_internal_links)} internal links")
    except Exception as e:
        st.warning(f"⚠️ Link generation failed: {e}")
        all_internal_links = []
    
    # Create a pool of links to distribute
    link_pool = list(all_internal_links)
    used_link_urls = set()

    # Generate intro if needed
    if any(t["level"] == "intro" for t in tokens):
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
        intro_md = generate_markdown(ps_intro.system, ps_intro.user, temperature=ps_intro.temperature).strip()
        if intro_md.startswith("#"):
            intro_md = intro_md.lstrip("# ").strip()
        parts.append(intro_md)
        previous_content = intro_md

    section_count = 0
    for t in tokens:
        if t["level"] == "intro":
            continue
        if t["level"].lower() == "shortcode":
            shortcode = (offer_row or {}).get("shortcode", "") if offer_row else ""
            parts.append(shortcode.strip() if shortcode.strip() else render_offer_block(offer_row or {}))
            continue
        
        heading = t["title"] or ""
        heading_lower = heading.lower()
        if "sign up" in heading_lower and "how to" in heading_lower:
            keyword = ss.get("keyword", "").strip() or ss.get("user_title", "").strip()
            date_str = today_long("US/Eastern")
            heading = f"How to Sign Up for the {keyword} on {date_str}" if keyword else f"How to Sign Up on {date_str}"
        
        section_count += 1
        
        if "overview" in heading_lower:
            objective = "Write about WHY this offer is valuable. Focus on appeal. 3-4 sentences max."
        elif "sign up" in heading_lower and "how to" in heading_lower:
            objective = "CRITICAL: Output ONLY a numbered list (1-5). NO PARAGRAPHS."
        elif "claim" in heading_lower or "how to" in heading_lower:
            objective = f"Provide WORKED EXAMPLE using: {event_context}. First-person, show win/loss scenarios."
        elif "eligibility" in heading_lower or "where" in heading_lower:
            objective = "Briefly state who qualifies. 2-3 sentences."
        elif "terms" in heading_lower or "details" in heading_lower:
            objective = "Cover fine print: odds, expiration, wagering. 3-4 sentences."
        else:
            objective = f"Write section under '{heading}'."
        
        brief = SectionBrief(
            section_id=f"sec-{uuid.uuid4().hex[:6]}",
            objective=objective,
            audience="Beginner—intermediate US sports bettors",
            constraints={},
            facts_and_points=[f"Featured game: {event_context}"] if event_context and ("claim" in heading_lower or "how to" in heading_lower) else [],
            retrieved_snippets=[],
        )
        
        query_parts = [heading, "sports betting promo", (offer_row or {}).get('brand', '')]
        query = " ".join(p for p in query_parts if p.strip())
        
        try:
            _hits = query_articles(query, k=3 if section_count > 2 else 8, snippet_chars=300 if section_count > 2 else 400)
            relevant = [h for h in _hits if h.get('score', 0) > 0.35][:5]
            snips = [(h.get("snippet") or "").strip() for h in relevant]
            brief.retrieved_snippets = [{"snippet": s} for s in snips if s]
        except Exception:
            pass

        # Distribute links from the pool (2-3 per section)
        section_links = []
        links_per_section = 2 if section_count <= 2 else 3
        
        while len(section_links) < links_per_section and link_pool:
            next_link = link_pool.pop(0)
            if next_link.url not in used_link_urls:
                section_links.append(next_link)
                used_link_urls.add(next_link.url)
        
        ps = make_promptsect(brief, offer_row or {}, section_links, disclaimer_for_state(state),
                            previous_content=previous_content[-1500:], available_states=available_states)
        body_md = generate_markdown(ps.system, ps.user, temperature=ps.temperature).strip()
        
        level = t["level"].lower()
        prefix = "##" if level == "h2" else "###" if level == "h3" else "####"
        section_text = f"{prefix} {heading}\n\n{body_md}"
        parts.append(section_text)
        previous_content += f"\n\n{section_text}"

    full_article = "\n\n".join(parts).strip()

    if switchboard_url and offer_row:
        from src.switchboard_links import inject_switchboard_links
        brand = (offer_row.get("brand") or "").strip()
        bonus_code = (offer_row.get("bonus_code") or "").strip()
        if brand and bonus_code:
            full_article = inject_switchboard_links(full_article, brand, bonus_code, switchboard_url, max_links=12)

    terms_section = _build_terms_section(offer_row, state)
    if terms_section:
        full_article = f"{full_article}\n\n{terms_section}"
    
    # CHANGE 2: Convert markdown to HTML
    st.info("🔄 Converting markdown to HTML...")
    html_article = markdown.markdown(
        full_article,
        extensions=['extra', 'nl2br', 'sane_lists'],
        output_format='html5'
    )
    
    return html_article


# =============================================================================
# COMPACT MAIN UI - ALL INPUTS IN ONE CONTAINER
# =============================================================================

with st.container():
    # Row 1: Offer + Refresh
    col1, col_refresh = st.columns([11, 1])
    
    with col1:
        offers_df = get_offers_df_cached()
        if offers_df.empty:
            st.error("No offers loaded. Check config.")
            st.stop()
        
        # Filter out casino offers
        opt_pairs = []
        for _, r in offers_df.iterrows():
            affiliate_offer = str(r.get("affiliate_offer", "")).strip().lower()
            offer_text = str(r.get("offer_text", "")).strip().lower()
            if any(kw in text for text in [affiliate_offer, offer_text] for kw in ["casino", "slots"]):
                continue
            
            brand_txt = str(r.get("brand", "")).strip()
            code_txt = str(r.get("bonus_code", "")).strip()
            main = str(r.get("affiliate_offer", "")).strip() or str(r.get("offer_text", "")).strip()
            label = f"{main} [{brand_txt}]" + (f" ({code_txt})" if code_txt else "")
            opt_pairs.append((str(r["offer_id"]), label))
        
        sel_idx = st.selectbox("Offer", options=list(range(len(opt_pairs))), 
                               format_func=lambda i: opt_pairs[i][1], key="offer_idx")
        offer_row = get_offer_by_id(offers_df, opt_pairs[sel_idx][0])
    
    with col_refresh:
        st.write("")
        if st.button("🔄", help="Refresh"):
            try:
                get_offers_df_cached.clear()
            except:
                pass
            if os.path.exists("data/offers_cache.pkl"):
                os.remove("data/offers_cache.pkl")
            st.rerun()
    
    # Compact offer info
    states = offer_row.get("states_list", []) or []
    code = (offer_row.get("bonus_code") or "").strip()
    link = (offer_row.get("switchboard_link") or "").strip()
    
    st.caption(f"**States:** {', '.join(states[:6]) if states and states != ['ALL'] else 'Nationwide'}  |  "
               f"**Code:** `{code if code else '—'}`  |  "
               f"**Link:** {'✅' if link else '⚠️'}")
    
    ss["switchboard_url"] = link or "https://switchboard.actionnetwork.com/offers?affiliateId=174"
    
    # Row 2: Title + Keyword + Sport + Date
    col_title, col_keyword, col_sport, col_date = st.columns([3, 2, 1, 1.5])
    
    sport_options = {"NFL": "nfl", "NBA": "nba", "MLB": "mlb", "NHL": "nhl"}
    
    with col_sport:
        sport_label = st.selectbox("Sport", list(sport_options.keys()))
        sport_selected = sport_options[sport_label]
    
    with col_date:
        target_date = st.date_input("Date", value=date.today(), 
                                     min_value=date.today(), max_value=date.today() + timedelta(7))
    
    # Fetch games
    target_datetime = datetime.combine(target_date, datetime.min.time()).replace(tzinfo=ZoneInfo("America/New_York"))
    event_context = ""
    selected_game = None
    
    try:
        games = get_games_for_date(sport_selected, target_datetime)
        if games:
            prime = filter_prime_time_games(games)
            default_game = prime[0] if prime else games[0]
            default_idx = games.index(default_game) if default_game in games else 0
            
            # Game selector
            game_options = [format_game_for_dropdown(g) for g in games]
            selected_idx = st.selectbox(
                f"Game ({len(games)} available)",
                options=list(range(len(games))),
                format_func=lambda i: game_options[i],
                index=default_idx
            )
            selected_game = games[selected_idx]
            event_context = format_event_for_prompt(selected_game, target_datetime)
            
            # Show what's selected
            is_prime = selected_game in prime if prime else False
            st.caption(f"📅 {event_context} {'⭐' if is_prime else ''}")
        else:
            st.info(f"ℹ️ No {sport_label} games on {target_date.strftime('%A, %b %d')}")
    except Exception as e:
        st.warning(f"⚠️ Could not fetch games: {e}")
    
    default_title = _build_default_title(offer_row, sport=sport_selected, event_context=event_context)
    
    with col_title:
        ss["user_title"] = st.text_input("Title", value=ss.get("user_title") or default_title)
    
    with col_keyword:
        ss["keyword"] = st.text_input("Keyword", value=ss.get("keyword", ""), 
                                      placeholder="e.g., BetMGM promo")
    
    # Competitor URLs (optional, collapsed)
    with st.expander("Competitor URLs (optional)"):
        comp_urls_txt = st.text_area("One per line", height=60, label_visibility="collapsed")

# Generate Outline button
if st.button("Generate Outline", type="primary"):
    comp_parts = [f"URL: {url}\n{scrape_competitor(url.strip())[:1500]}" 
                  for url in comp_urls_txt.splitlines() if url.strip()]
    comp_text = "\n\n".join(comp_parts)[:6000]
    
    tokens = ai_outline_tokens_rag(
        keyword=ss["keyword"].strip() or ss["user_title"].strip(),
        exact_title=ss["user_title"].strip(),
        brand=offer_row.get("brand", ""),
        offer_text=offer_row.get("offer_text", ""),
        page_type_hint=(offer_row.get("page_type") or "").strip(),
        comp_urls_txt=comp_text,
    )
    ss["tokens_cached"] = tokens
    ss["box_mode"] = "outline"
    ss["pending_article_box"] = tokens_to_outline_text(tokens)
    st.success("✅ Outline ready")
    st.rerun()

# Editor with tabs
st.divider()
ss.setdefault("article_box", "")
if "pending_article_box" in ss:
    ss["article_box"] = ss.pop("pending_article_box")

tab_code, tab_preview = st.tabs(["📝 Code Editor", "👁️ Preview"])

with tab_code:
    if ss["box_mode"] == "outline":
        hint = "📍 Outline: [H2: Title], [H3: Subtitle], [INTRO], [SHORTCODE]"
        st.caption(hint)
        st.text_area("Outline", key="article_box", height=400, label_visibility="collapsed")
    else:
        hint = "📄 Edit HTML directly. Changes will reflect in Preview tab."
        st.caption(hint)
        st.text_area("HTML Code", key="article_box", height=400, label_visibility="collapsed")

with tab_preview:
    if ss["box_mode"] == "outline":
        st.info("📐 Generate draft to see preview")
    else:
        content = ss.get("article_box", "")
        if content:
            preview_html = f"""
            <style>
            .preview-container {{
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
                line-height: 1.6;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }}
            .preview-container h1 {{
                font-size: 2.5em;
                margin-bottom: 0.5em;
                color: #1a1a1a;
            }}
            .preview-container h2 {{
                font-size: 2em;
                margin-top: 1.5em;
                border-bottom: 2px solid #eee;
                padding-bottom: 0.3em;
                color: #2a2a2a;
            }}
            .preview-container h3 {{
                font-size: 1.5em;
                margin-top: 1.2em;
                color: #3a3a3a;
            }}
            .preview-container p {{
                margin-bottom: 1em;
            }}
            .preview-container a {{
                color: #0066cc;
                text-decoration: none;
            }}
            .preview-container a:hover {{
                text-decoration: underline;
            }}
            .preview-container code {{
                background: #f4f4f4;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'Courier New', monospace;
            }}
            .preview-container ul, .preview-container ol {{
                margin-left: 1.5em;
                margin-bottom: 1em;
            }}
            .preview-container li {{
                margin-bottom: 0.5em;
            }}
            </style>
            <div class="preview-container">
            {content}
            </div>
            """
            st.html(preview_html)
        else:
            st.info("No content to preview")

# Actions row
col_gen, col_fmt, col_name, col_dl = st.columns([2, 1.5, 2, 1])

with col_gen:
    gen_label = "Generate Draft" if ss["box_mode"] == "outline" else "Regenerate"
    if st.button(gen_label, type="primary", use_container_width=True):
        if ss["box_mode"] == "outline":
            tokens = parse_outline_tokens(ss["article_box"])
            if tokens:
                ss["tokens_cached"] = tokens
        
        tokens = ss.get("tokens_cached", [])
        if tokens:
            with st.spinner("Generating..."):
                full_md = generate_article_from_tokens(
                    tokens=tokens, title=ss["user_title"].strip(), offer_row=offer_row,
                    state="ALL", event_context=event_context, sport=sport_selected,
                    switchboard_url=ss.get("switchboard_url", ""), target_date=target_datetime,
                )
                ss["box_mode"] = "draft"
                ss["pending_article_box"] = full_md
                st.success("✅ Done")
                st.rerun()

with col_fmt:
    export_fmt = st.selectbox("Format", ["Markdown", "HTML", "Word"], label_visibility="collapsed")

with col_name:
    export_name = st.text_input("Filename", "article", label_visibility="collapsed")

with col_dl:
    content = ss.get("article_box", "").strip()
    if export_fmt == "Markdown":
        data = content.encode("utf-8")
        ext = ".md"
    elif export_fmt == "HTML":
        data = f"<article>\n{content.replace(chr(10), '<br>')}\n</article>".encode("utf-8")
        ext = ".html"
    else:
        from docx import Document
        buf = io.BytesIO()
        doc = Document()
        for p in content.split("\n\n"):
            doc.add_paragraph(p)
        doc.save(buf)
        data = buf.getvalue()
        ext = ".docx"
    
    st.download_button("📥 Export", data=data, file_name=f"{export_name or 'article'}{ext}",
                       disabled=not content, use_container_width=True)