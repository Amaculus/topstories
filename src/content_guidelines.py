# src/content_guidelines.py
"""
Content guidelines for sports betting promo articles.
Defines tone, compliance rules, and style constraints.
"""

GUIDELINES = {
    "core_principles": {
        "compliance_first": {
            "description": "Maintain strict compliance with US gambling regulations. Never imply guaranteed outcomes.",
            "prohibited_claims": [
                "risk-free",
                "guaranteed win",
                "easy money",
                "sure thing",
                "can't lose",
                "surefire"
            ],
        },
        "authentic_tone": {
            "description": "Write like an informed person having a genuine conversation - not a promotional advertisement.",
            "avoid_phrases": [
                "FABULOUS APP! FUN AND ENGAGING!",
                "Experience the thrill like never before!",
                "Revolutionary gaming experience",
                "Premier online sports betting platform",
                "With generous bonuses, a user-friendly app, and a commitment to responsible gambling",
            ],
            "overused_words": [
                "premier", "generous", "solid choice", "stands out",
                "commitment to", "user-friendly", "exciting", "amazing",
                "incredible", "outstanding", "exceptional", "revolutionary"
            ],
        },
    },
    "tone": {
        "voice": "conversational",
        "perspective": "second_person",
        "formality": "casual_informative",
        "max_sentence_length": 25,  # words
        "use_contractions": True,
        "avoid_jargon": True,
    },
    "content_structure": {
        "paragraph_length": "2-4 sentences, 40-70 words",
        "sentence_variety": "Mix short punchy sentences (8-12 words) with medium compound sentences (15-25 words)",
        "pacing": "Front-load key info (offer amount, code, eligibility), details later",
        "list_usage": "Minimal - prefer natural paragraph flow",
    },
    "compliance_requirements": {
        "state_specificity": "Mention state restrictions when applicable",
        "terms_transparency": "Link to or mention full terms",
    },
}


def get_style_instructions() -> str:
    """Return consolidated style instructions for prompts."""
    return """STYLE GUIDE (Top Stories - Sports Betting Promo):

VOICE & TONE:
- Conversational and informative, like a knowledgeable friend sharing a deal
- Casual but professional - use contractions naturally
- Excited but not overselling - avoid hyperbolic marketing language
- Honest about limitations, clear about requirements

FORBIDDEN PHRASES (never use):
- "risk-free" (except in official bonus name like "risk-free bet credit")
- "guaranteed win", "can't lose", "sure thing", "easy money"
- "revolutionary", "premier", "exceptional", "stands out as"
- "generous bonuses and user-friendly app" (overused cliché)
- Marketing hype like "experience the thrill like never before"

SECTION VARIETY (critical - avoid repetition):
- Each section should ADD new information, not restate previous sections
- If the intro mentioned the states, don't list them again in every section
- If you explained the mechanic in Overview, don't re-explain it in Eligibility
- Later sections should be SHORTER and more specific
- Use varied sentence structures - not every section starts with "To..."

SECTION-SPECIFIC GUIDANCE:
- Overview: Why this offer matters, what makes it valuable
- How to Claim: Worked example with dollar amounts and outcomes
- Eligibility: Who qualifies (brief) - skip restating the offer
- Terms: Fine print only - odds requirements, expirations, restrictions
- Responsible Gaming: 2-3 sentences max with helpline

SENTENCE STRUCTURE:
- Mix short (8-12 words) and medium (15-25 words) sentences
- Max 25 words per sentence
- Use contractions (don't, you'll, it's) naturally
- Vary rhythm - don't start every sentence the same way

PARAGRAPH FLOW:
- 2-4 sentences per paragraph, 40-70 words total
- Front-load important info (offer amount, promo code, key dates)
- Details and fine print come later
- Natural flow over rigid list formatting

VOCABULARY:
- Beginner-friendly - explain betting terms inline if needed
- Say "bet" not "wager" (more natural)
- Avoid marketing jargon and clichés
- Be specific: "$150 in bonus bets" not "generous bonus"

COMPLIANCE (non-negotiable):
- Always mention 21+ age requirement
- Include responsible gaming helpline
- State-specific restrictions when applicable
- Never imply guaranteed outcomes
- No "risk-free" claims (unless quoting official bonus name)"""


def get_prohibited_patterns() -> list[str]:
    """Regex patterns for compliance checking."""
    return [
        r"\bguarantee(d)?\b(?! applies)",  # "guaranteed" unless "guarantee applies"
        r"\bsurefire\b",
        r"\bcan'?t lose\b",
        r"\beasy money\b",
        r"\brisk[-\s]?free\b(?! bet credit)",  # Allow "risk-free bet credit"
        r"\bpremier (?:online )?(?:sports )?betting platform\b",
        r"\bstands out as\b",
        r"\bcommitment to responsible gambling\b",  # Cliché phrasing
    ]


def get_temperature_by_section(section_type: str) -> float:
    """Return appropriate temperature for different content types."""
    temps = {
        "intro": 0.7,      # More natural variation
        "h2": 0.5,         # Balanced creativity
        "h3": 0.4,         # Still varied
        "outline": 0.6,    # More creative
    }
    return temps.get(section_type, 0.5)  # Default higher