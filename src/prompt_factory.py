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

def extract_common_phrases(text: str) -> list[str]:
    """Extract common filler phrases that should be avoided in subsequent sections."""
    if not text:
        return []
    
    # Patterns for common repetitive structures
    patterns = [
        r"To (?:qualify|claim|get|take advantage|access|receive|sign up) (?:for|this|the) [\w\s]{1,30}",
        r"In order to [\w\s]{1,30}",
        r"(?:This|The) (?:offer|promo|bonus) (?:is|allows|gives|provides) [\w\s]{1,30}",
        r"(?:New|Eligible) (?:users|customers|bettors) can [\w\s]{1,30}",
        r"available (?:to|for) (?:new|eligible) [\w\s]{1,30}",
    ]
    
    found = []
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        found.extend([m.strip() for m in matches if len(m.strip()) > 10])
    
    # Deduplicate and return unique phrases
    return list(set(found))[:5]  # Limit to top 5 to avoid overwhelming the prompt

# ---------- main factories ----------

def make_promptsect(
    brief: SectionBrief,
    offer_row: Dict[str, Any],
    inline_links: List[InlineLinkSpec] | List[Dict[str, Any]],
    disclaimer: str,
    allow_cta: bool = True,
    previous_content: str = "",
    available_states: list[str] = None,
    keyword: str = "",
    current_keyword_count: int = 0,
    target_keyword_total: int = 9,
) -> PromptSect:
    """
    Build a section-writing prompt (system+user) for a single H2/H3.
    """
    # Normalize offer bits
    brand = (offer_row.get("brand") or "").strip()
    offer_text = (offer_row.get("offer_text") or "").strip()
    bonus_code = (offer_row.get("bonus_code") or "").strip()
    focus_term = keyword.strip() if keyword else f"{brand} promo"
    
    # Extract critical offer details
    expiration_days = offer_row.get("bonus_expiration_days", 7)
    min_odds = offer_row.get("minimum_odds", "")
    wagering = offer_row.get("wagering_requirement", "")
    bonus_amount = offer_row.get("bonus_amount", "")

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
    is_numbered_list = "NUMBERED STEP-BY-STEP" in brief.objective or "numbered list" in brief.objective.lower()
    
    # Extract blacklisted phrases from previous content
    blacklisted_phrases = extract_common_phrases(previous_content) if previous_content else []
    constraints = brief.constraints or {}
    blacklisted_phrases.extend(constraints.get("avoid_phrases", []))
    blacklisted_phrases = list(set(blacklisted_phrases))[:8]  # Limit to 8 most important

    # Rules - adjust based on format
    if is_numbered_list:
        rules = [
            "YOU MUST OUTPUT A NUMBERED LIST.",
            "Format: 1. [sentence] 2. [sentence] 3. [sentence] 4. [sentence] 5. [sentence]",
            f"Include '{focus_term}' in at least 2 steps naturally.",
            "Use 1-2 internal links naturally within the steps.",
            "Explain each step thoroughly while respecting style guidelines",
            "Do NOT include an introduction or conclusion - ONLY the numbered list.",
            f"MANDATORY: Use the exact phrase '{keyword}' at least once in the list.",
        ]
        format_instruction = "OUTPUT FORMAT: Numbered list (1. 2. 3. 4. 5.) explaining each step thoroughly."
    else:
        rules = [
            "Begin directly under the heading with helpful, concise copy (no fluff).",
            "Match the STYLE of the examples, but use the FACTS from the SOURCE OF TRUTH section.",
            "MUST use at least 2-3 internal links. They are pre-selected for relevance. Use descriptive anchor text that flows naturally.",
            "Do NOT copy sentences from style examples - paraphrase and use your own phrasing.",
            "Vary your sentence openings - don't repeat previous patterns.",
            "Maintain neutral, compliant tone; avoid marketing hype and prohibited phrases.",
            "No tables, no HTML.",
            f"Use '{focus_term}' as the main reference, not the full offer name",
            "Do NOT print or restate the heading; write paragraphs only.",
            f"MANDATORY: Include the exact phrase '{keyword}' at least once naturally in this section's body text.",
        ]
        format_instruction = "OUTPUT FORMAT: 2-4 flowing paragraphs."
    
    if not allow_cta:
        rules.append("Do NOT include any CTA; a promo block is inserted elsewhere.")
    else:
        rules.append("Include at most one brief, natural CTA sentence if it serves the reader.")

    # System prompt with stronger fact-checking emphasis
    sys = (
        "You are an expert SEO content writer specializing in US sports betting promo announcements. "
        "Your content must be compliant, authentic, and match the house style exactly. "
        "Avoid marketing clichés, repetitive phrasing, and write like an informed person sharing useful information. "
        "\n\n"
        "CRITICAL INSTRUCTION: You must use ONLY the exact offer details provided in the SOURCE OF TRUTH section. "
        "NEVER invent odds, expiration dates, wagering requirements, or other terms. "
        "If a detail is not explicitly provided, say 'see full terms' or 'check the operator's website' instead of guessing. "
        "Making up facts damages credibility and violates compliance standards."
    )

    # Keyword tracking info
    keyword_progress = f"(Used {current_keyword_count}/{target_keyword_total} times so far)"
    keyword_needed = current_keyword_count < target_keyword_total

    # User prompt with SOURCE OF TRUTH section
    user = f"""WRITE UNDER THIS HEADING EXACTLY (DO NOT PRINT THE HEADING):
{heading_title}

{format_instruction}

OBJECTIVE:
{brief.objective}

AUDIENCE:
{brief.audience or "Beginner—intermediate US sports bettors (ages 21-65)"}

=== SOURCE OF TRUTH - DO NOT DEVIATE ===
These are EXACT facts from the offer sheet. Do NOT modify, invent, or approximate:

CORE OFFER DETAILS:
- Brand: {brand or "[not provided]"}
- Offer: {offer_text or "[not provided]"}
- Bonus Amount: {bonus_amount or "[not provided]"}
- Bonus Code: {bonus_code or "[not provided]"}
- Available in: {states_text}

CRITICAL TERMS (use these EXACT values if you mention them):
- Expiration: {expiration_days} days (say "expire in {expiration_days} days" - NOT 7, NOT 30, NOT 14 - exactly {expiration_days})
- Minimum Odds: {min_odds if min_odds else "[see terms - do not guess]"}
- Wagering: {wagering if wagering else "[see terms - do not guess]"}

RULE: If you mention expiration, odds requirements, or wagering rules, use ONLY the values above or say "check full terms". NEVER invent numbers.

=== END SOURCE OF TRUTH ===

{style_guide}

STYLE EXAMPLES - Match the tone, rhythm, and voice (NOT the facts):
{snippets_md}

^ These show how WE write about sports betting promos. Mirror:
- Sentence length/rhythm (mix short punchy + medium compound)
- Vocabulary choices and level of formality
- Pacing (front-load key info, details later)
- Natural conversational tone without marketing hype

BUT REMEMBER: Use the FACTS from SOURCE OF TRUTH, not from these examples.

ADDITIONAL FACTS (if relevant):
{facts_md}

INTERNAL LINKS (MUST use at least 2-3 of these):
{links_md}

CRITICAL: You MUST include at least 2 internal links in this section. These links are pre-selected for relevance and value. Use descriptive anchor text that flows naturally in your prose.

KEYWORD USAGE:
Your primary focus term is: "{keyword}" {keyword_progress}

Requirements:
- {"MUST use" if keyword_needed else "Should use"} the exact phrase "{keyword}" at least ONCE in this section
- Use it naturally in a sentence, not forced or awkward
- Good: "Sign up with {keyword} to unlock this offer"
- Bad: "The {keyword} {keyword} is available now" (repetitive)
- This is in addition to any internal link anchor text

PREVIOUSLY WRITTEN (DO NOT REPEAT - the reader already saw this):
{previous_content or "(this is the first section)"}

^ The intro already covered: the offer amount, bonus code, and states list.
DO NOT repeat these unless specifically relevant to this section's objective.
Your job is to ADD NEW INFORMATION specific to "{heading_title}", not restate the intro.

ANTI-REPETITION RULES (CRITICAL):
Review what you've already written in PREVIOUSLY WRITTEN content.

DO NOT repeat:
- Sentence structures you see there (if you see "To qualify for", don't use "To [verb] for")
- Opening phrases (if you see "This offer allows", don't use "This promo gives")  
- Key sentences or concepts already covered
- The same information from previous sections

{"PHRASES TO AVOID (already overused):" if blacklisted_phrases else ""}
{chr(10).join(f"❌ {phrase}" for phrase in blacklisted_phrases) if blacklisted_phrases else ""}

Start this section with a DIFFERENT structure than previous sections.
Vary your vocabulary - if previous sections used "qualify", try "eligible" or "meet requirements".
If previous sections opened with "To [verb]", open with a question, statement, or direct instruction instead.

PHRASING VARIETY REQUIREMENTS:
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

    return PromptSect(system=sys, user=user, rules=rules, temperature=0.4)

def make_intro_prompt(
    *,
    brand: str,
    offer_text: str,
    bonus_code: str,
    date_str: str,
    available_states: list[str],
    event_context: str = "",
    keyword: str = "",
    is_mo_launch: bool = False,
    bonus_expiration_days: int = 7,
    bonus_amount: str = "",
) -> PromptSect:
    """Build the lede/intro prompt with natural date and event integration."""
    sys = (
        "You are an expert sports betting news writer. "
        "Write a concise, engaging intro paragraph for a promo announcement article. "
        "Be factual and compliant - avoid marketing hype. "
        "The intro must be ONE flowing paragraph with natural transitions. "
        "\n\n"
        "CRITICAL: Use ONLY the exact offer details provided. Never invent expiration dates, "
        "odds requirements, or other terms. If not provided, omit rather than guess."
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

    # Build context-aware instructions for Missouri launch
    if is_mo_launch:
        context_instruction = f"""
Write a natural 3-4 sentence intro paragraph for this Missouri sports betting launch promo.

CONTEXT:
- Missouri Launch Timeline: Registration opens November 17, 2024. Full sports betting launches December 1, 2024.
- Offer: {offer_text}
- Bonus Code: {bonus_code}
- State: Missouri only

Your intro should:
- Hook with the launch timing and offer value
- Mention both key dates (registration Nov 17, full launch Dec 1)
- Reference the bonus code twice naturally
- Emphasize this is for the Missouri launch

Write as ONE flowing paragraph. Be conversational - like breaking news about an exciting opportunity.

GOOD EXAMPLE (match this natural style):
Missouri residents can sign up for sports betting starting November 17 with the DraftKings Missouri promo code to claim $300 in bonus bets ahead of the state's full launch on December 1. Register early with the DraftKings Missouri promo code BETACTION and place a $5 bet to receive $300 in bonus bets instantly. This Missouri-only offer allows new users to get a head start on sports betting before the official December 1 launch date.

Notice how it:
- Leads with the launch timeline
- Integrates both dates naturally
- Mentions the code twice without repetition
- Focuses on the launch opportunity
"""
    elif event_context:
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
        "List ALL states explicitly if multiple states" if not is_mo_launch else "Focus on Missouri launch timing",
        "NO exclamation points anywhere",
        "Don't overuse contractions. Use them naturally. (you will, not you'll)",
        f"Use '{focus_term}' as the main reference, not the full offer name",
        "Sound conversational but professional",
        f"CRITICAL: If mentioning expiration, say 'expire in {bonus_expiration_days} days' - no other number",
    ]

    user = f"""Write a 3-4 sentence intro paragraph following this structure:

{context_instruction}

=== SOURCE OF TRUTH - DO NOT DEVIATE ===
- Brand: {brand}
- Offer: {offer_text}
- Bonus Amount: {bonus_amount if bonus_amount else "[use amount from offer text]"}
- Bonus code: {bonus_code}
- States: {states_list}
- Expiration: {bonus_expiration_days} days (if you mention it, say "{bonus_expiration_days} days" exactly)
{"- Launch Type: Missouri registration (Nov 17) and full launch (Dec 1)" if is_mo_launch else ""}

CRITICAL: Use these exact values. Do NOT invent or modify any numbers.
=== END SOURCE OF TRUTH ===

{style_guide}

CRITICAL FORMAT REQUIREMENTS:
- Output ONE flowing paragraph
- NO quotation marks around sentences
- NO separating sentences into quoted blocks
- Natural transitions between sentences
{"- Emphasize Missouri launch timeline" if is_mo_launch else "- Integrate the date naturally (e.g., 'today, " + date_str + "' or 'ahead of [event]')"}

RULES:
- """ + "\n- ".join(rules)

    return PromptSect(system=sys, user=user, rules=rules, temperature=0.6)