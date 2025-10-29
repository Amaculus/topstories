# src/prompt_factory.py
from __future__ import annotations
import re, json
from typing import List, Dict, Any, Iterable
from src.models import SectionBrief, InlineLinkSpec, PromptSect
from src.offers_layer import render_offer_block
from src.content_guidelines import get_style_instructions

# ---------- helpers ----------
def _format_instructions_html() -> str:
    """Return HTML-specific formatting instructions."""
    return """
HTML OUTPUT REQUIREMENTS:
- Output ONLY raw HTML - NO markdown syntax
- Use semantic HTML tags: <h2>, <h3>, <p>, <a>, <ul>, <ol>, <li>, <strong>, <em>
- Paragraphs: <p>Your text here</p>
- Links: <a href="url">anchor text</a>
- Lists: <ol><li>Item 1</li><li>Item 2</li></ol>
- Bold: <strong>text</strong>
- Emphasis: <em>text</em>
- NO markdown formatting (##, **, [], etc.)
- NO code fences or backticks
- Clean, semantic HTML only
"""

def _coalesce_text(x: Any) -> str:
    """Convert a snippet object (dict|str|other) to a clean string."""
    if isinstance(x, str):
        s = x
    elif isinstance(x, dict):
        s = x.get("snippet") or x.get("text") or x.get("content") or x.get("preview") or ""
    else:
        s = str(x or "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _links_md(inline_links: Iterable[InlineLinkSpec | Dict[str, Any]]) -> str:
    """Render inline-link suggestions into markdown bullets."""
    lines: List[str] = []
    for it in inline_links or []:
        if isinstance(it, InlineLinkSpec):
            title, url, rec = it.title, it.url, it.recommended_anchors or []
        elif isinstance(it, dict):
            title = it.get("title") or it.get("anchor") or it.get("text") or ""
            url   = it.get("url") or it.get("href") or ""
            rec   = it.get("recommended_anchors") or []
        else:
            title, url, rec = str(it), "", []
        if not (title or url):
            continue
        anchor_hint = f" — anchors: {', '.join(rec[:3])}" if rec else ""
        disp = f"[{title}]({url})" if url else title
        lines.append(f"- {disp}{anchor_hint}")
    return "\n".join(lines) or "(none)"

def _extract_heading_from_objective(obj_text: str) -> str:
    """Grab the heading title from '... EXACTLY titled 'Heading' ...' if present."""
    m = re.search(r"titled\s+'([^']+)'", obj_text or "", flags=re.I)
    return (m.group(1).strip() if m else "").strip()

# ---------- main factories ----------

# src/prompt_factory.py - make_promptsect function

def make_promptsect(
    brief: SectionBrief,
    offer_row: Dict[str, Any],
    inline_links: List[InlineLinkSpec] | List[Dict[str, Any]],
    disclaimer: str,
    allow_cta: bool = True,
    previous_content: str = "",
    available_states: list[str] = None,
    keyword: str = "",
) -> PromptSect:
    """
    Build a section-writing prompt (system+user) for a single H2/H3.
    """
    # Normalize offer bits
    brand = (offer_row.get("brand") or "").strip()
    offer_text = (offer_row.get("offer_text") or "").strip()
    bonus_code = (offer_row.get("bonus_code") or "").strip()
    focus_term = keyword.strip() if keyword else f"{brand} promo"


    # Normalize snippets
    snips_list = [t for t in (_coalesce_text(s) for s in (brief.retrieved_snippets or [])) if t]
    snippets_md = "\n\n".join(snips_list) or "(none)"

    # Normalize facts/points
    facts_list = [f for f in (brief.facts_and_points or []) if isinstance(f, str) and f.strip()]
    facts_md = "\n".join(f"- {f.strip()}" for f in facts_list) or "(none)"

    # Links
    links_md = _links_md(inline_links)

    import os
    if os.getenv("DEBUG"):
        import streamlit as st
        print(f"**Links being added to prompt:**")
        print(links_md)
    # Heading
    heading_title = _extract_heading_from_objective(brief.objective) or "Section"

    # Format states
    if not available_states or "ALL" in available_states:
        states_text = "multiple states"
    elif len(available_states) == 1:
        states_text = f"{available_states[0]} only"
    else:
        states_text = ", ".join(available_states)

    # Get style instructions
    style_guide = get_style_instructions()

    # Detect if this is a numbered list section
    is_numbered_list = "NUMBERED STEP-BY-STEP" in brief.objective

    # Rules - adjust based on format
    if is_numbered_list:
        rules = [
            "YOU MUST OUTPUT A NUMBERED LIST.",
            "Format: 1. [sentence] 2. [sentence] 3. [sentence] 4. [sentence] 5. [sentence]",
            f"Mention '{brand} bonus code {bonus_code}' in at least 2 steps.",
            "Use 1-2 internal links naturally within the steps.",
            "Explain each step thoroughly while respecting style guidelines",
            "Do NOT include an introduction or conclusion - ONLY the numbered list.",
        ]
        format_instruction = "OUTPUT FORMAT: Numbered list (1. 2. 3. 4. 5.) explaining each step thoroughly."
    else:
        rules = [
            "Begin directly under the heading with helpful, concise copy (no fluff).",
            "Match the STYLE of the examples, but use the FACTS from the promo details.",
            "MUST use at least 2-3 internal links. They are pre-selected for relevance, Use descriptive anchor text that flows naturally in your prose",
            "Do NOT copy sentences from style examples - paraphrase and use your own phrasing.",
            "Vary your sentence openings - don't repeat previous patterns.",
            "Maintain neutral, compliant tone; avoid marketing hype and prohibited phrases.",
            "No tables, no HTML.",
            f"Use '{focus_term}' as the main reference, not the full offer name",
            "Do NOT print or restate the heading; write paragraphs only.",
        ]
        format_instruction = "OUTPUT FORMAT: 2-4 flowing paragraphs."
    
    if not allow_cta:
        rules.append("Do NOT include any CTA; a promo block is inserted elsewhere.")
    else:
        rules.append("Include at most one brief, natural CTA sentence if it serves the reader.")

    # System prompt
    sys = (
        "You are an expert SEO content writer specializing in US sports betting promo announcements. "
        "Your content must be compliant, authentic, and match the house style exactly. "
        "Avoid marketing clichés, repetitive phrasing, and write like an informed person sharing useful information."
    )

    # User prompt
    user = f"""WRITE UNDER THIS HEADING EXACTLY (DO NOT PRINT THE HEADING):
{heading_title}

{format_instruction}

OBJECTIVE:
{brief.objective}

AUDIENCE:
{brief.audience or "Beginner—intermediate US sports bettors (ages 21-65)"}

{style_guide}

STYLE EXAMPLES - Match the tone, rhythm, and voice (NOT the facts):
{snippets_md}

^ These show how WE write about sports betting promos. Mirror:
- Sentence length/rhythm (mix short punchy + medium compound)
- Vocabulary choices and level of formality
- Pacing (front-load key info, details later)
- Natural conversational tone without marketing hype

PROMO FACTS (use these exact details - SOURCE OF TRUTH):
- Brand: {brand or "(none)"}
- Offer: {offer_text or "(none)"}
- Bonus code: {bonus_code or "(none)"}
- Available in: {states_text}

ADDITIONAL FACTS (if relevant):
{facts_md}

INTERNAL LINKS (MUST use at least 2-3 of these):
{links_md}

CRITICAL: You MUST include at least 2 internal links in this section. These links are pre-selected for relevance and value. Use descriptive anchor text that flows naturally in your prose.

PREVIOUSLY WRITTEN (DO NOT REPEAT - the reader already saw this):
{previous_content or "(this is the first section)"}

^ The intro already covered: the offer amount, bonus code, and states list.
DO NOT repeat these unless specifically relevant to this section's objective.
Your job is to ADD NEW INFORMATION specific to "{heading_title}", not restate the intro.

PHRASING VARIETY REQUIREMENTS (CRITICAL):
- Do NOT start with "To take advantage of..." or "To qualify for..." or "To get started with..."
- Vary your openings: use questions, statements, direct instructions, examples
- Good openings: "Here's how it works:" / "Claiming this is simple:" / "You'll need..." / "Start by..." / "For example, if I..."
- Never repeat the same sentence structure from PREVIOUSLY WRITTEN content
- Mix short (8-12 words) and medium (15-25 words) sentences

CRITICAL ANTI-REPETITION RULES:
- If previous content mentions the states list, DO NOT list all states again
- If previous content explains the basic offer, DO NOT re-explain it
- Focus on the SPECIFIC purpose of this section (see OBJECTIVE above)
- Later sections should be SHORTER (2-4 sentences) and more focused

DISCLAIMER (append if required):
{(disclaimer or "").strip() or "(none)"}

RULES:
- """ + "\n- ".join(rules)

    return PromptSect(system=sys, user=user, rules=rules, temperature=0.5)

# src/prompt_factory.py - make_intro_prompt function

def make_intro_prompt(
    *,
    brand: str,
    offer_text: str,
    bonus_code: str,
    date_str: str,
    available_states: list[str],
    event_context: str = "",
    keyword: str = "",  # ADD THIS PARAMETER
) -> PromptSect:
    """Build the lede/intro prompt with natural date and event integration."""
    sys = (
        "You are an expert sports betting news writer. "
        "Write a concise, engaging intro paragraph for a promo announcement article. "
        "Be factual and compliant - avoid marketing hype. "
        "The intro must be ONE flowing paragraph with natural transitions."
    )

    focus_term = keyword.strip() if keyword else f"{brand} promo"
    brand = (brand or "").strip()
    offer_text = (offer_text or "").strip()
    bonus_code = (bonus_code or "").strip()
    event_context = (event_context or "").strip()

    # Format states
    if not available_states or "ALL" in available_states:
        states_list = "multiple states"
        states_sentence = "This promo is available to new users nationwide."
    elif len(available_states) == 1:
        states_list = f"{available_states[0]} only"
        states_sentence = f"This {available_states[0]}-only offer is available now."
    else:
        states_str = ", ".join(available_states)
        states_list = states_str
        states_sentence = f"This promo is available to new users in {states_str}."

    style_guide = get_style_instructions()

# Build context-aware instructions
    if event_context:
        context_instruction = f"""
    Write a natural 3-4 sentence intro paragraph for this {brand} promo.

    CONTEXT:
    - Featured Event: {event_context}
    - Date: {date_str}
    - Offer: {offer_text}
    - Bonus Code: {bonus_code}
    - States: {states_sentence}

    Your intro should:
    - Hook with the offer value and tie it to the upcoming event naturally
    - Mention the bonus code twice in a way that flows
    - Include the date/event context organically (don't force it)
    - State which users are eligible

    Write as ONE flowing paragraph. Be conversational - like you're texting a friend about a good deal.

    GOOD EXAMPLE (match this natural style):
    Unlock $200 in bonus bets by signing up with the bet365 bonus code TOPACTION ahead of Monday Night Football tonight. Register with the bet365 bonus code TOPACTION to place a $5 bet and receive $200 in bonus bets, win or lose. This promo is available to new users in AZ, CO, IA, IL, IN, KS, KY, LA, MD, NC, NJ, OH, PA, TN, VA. Claim this bet365 promo to bet the Chiefs vs. Jaguars on Monday Night Football at 8:15 p.m. ET on ESPN.

    Notice how it:
    - Leads with value ("Unlock $200")
    - Connects to the event naturally ("ahead of Monday Night Football tonight")
    - Mentions the code twice without sounding repetitive
    - States eligibility clearly but doesn't belabor it
    - Sounds like useful information, not a sales pitch
    """
    else:
        context_instruction = f"""
    Write a natural 3-4 sentence intro paragraph for this {brand} promo.

    CONTEXT:
    - Date: {date_str}
    - Offer: {offer_text}
    - Bonus Code: {bonus_code}
    - States: {states_sentence}

    Your intro should:
    - Hook with the offer value
    - Mention the bonus code twice naturally
    - Include today's date in a conversational way
    - State which users are eligible

    Write as ONE flowing paragraph. Be conversational - like you're texting a friend about a good deal.

    GOOD EXAMPLE (match this natural style):
    New FanDuel users can claim up to $200 in bonus bets with the FanDuel promo code today, October 14, 2025. Sign up with the FanDuel promo code to place a $5+ bet and receive $200 in bonus bets instantly. This offer is available to new customers in Arizona, Colorado, and 15 other states nationwide.

    Notice how it:
    - Leads with value ("claim up to $200")
    - Includes the date naturally ("today, October 14, 2025")
    - Mentions the code twice without repetition
    - Keeps it brief and useful, not hypey
    """

    rules = [
        "Write as ONE flowing paragraph (not separate quoted sentences)",
        "Use exact dollar amounts and facts from the promo details",
        f"Mention '{brand} bonus code {bonus_code}' twice naturally",
        "Be conversational and useful - avoid marketing hype",
        "List ALL states explicitly if multiple states",
        "NO exclamation points anywhere",
        "Don't overuse contractions. Use them naturally. (you will, not you'll)",
        f"Use '{focus_term}' as the main reference, not the full offer name",
        "Sound conversational but professional",
    ]

    user = f"""Write a 3-4 sentence intro paragraph following this structure:

{context_instruction}

PROMO DETAILS (use exact amounts and facts):
- Brand: {brand}
- Offer: {offer_text}
- Bonus code: {bonus_code}
- States: {states_list}

{style_guide}

CRITICAL FORMAT REQUIREMENTS:
- Output ONE flowing paragraph
- NO quotation marks around sentences
- NO separating sentences into quoted blocks
- Natural transitions between sentences
- Integrate the date naturally (e.g., "today, {date_str}" or "ahead of [event]")

GOOD EXAMPLE (one flowing paragraph):
Unlock $200 in bonus bets by signing up with the bet365 bonus code TOPACTION ahead of Monday Night Football tonight. Register with the bet365 bonus code TOPACTION to place a $5 bet and receive $200 in bonus bets, win or lose. This promo is available to new users in AZ, CO, IA, IL, IN, KS, KY, LA, MD, NC, NJ, OH, PA, TN, VA. Claim this bet365 promo to bet the Chiefs vs. Jaguars on Monday Night Football at 8:15 p.m. ET on ESPN.


RULES:
- """ + "\n- ".join(rules)

    return PromptSect(system=sys, user=user, rules=rules, temperature=0.6)