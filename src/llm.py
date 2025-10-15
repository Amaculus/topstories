# src/llm.py
import os
from typing import Optional
from openai import OpenAI
from openai import APIError, APIConnectionError, RateLimitError

_client_instance: Optional[OpenAI] = None

def _get_client() -> OpenAI:
    global _client_instance
    if _client_instance is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set in your environment (.env).")
        _client_instance = OpenAI(api_key=api_key)
    return _client_instance

def generate_markdown(
    system: str, 
    user: str, 
    model: Optional[str] = None, 
    temperature: float = 0.3
) -> str:
    """
    Call the chat model and return Markdown text.
    
    Args:
        system: System prompt text
        user: User prompt text
        model: Override model name (defaults to env MODEL_NAME or 'gpt-4o-mini')
        temperature: Sampling temperature (0.0-2.0). Lower = more deterministic.
    """
    client = _get_client()
    model_name = model or os.getenv("MODEL_NAME", "gpt-4o-mini")

    try:
        resp = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
    except (APIConnectionError, RateLimitError, APIError) as e:
        raise RuntimeError(f"OpenAI API error: {e}") from e

    content = resp.choices[0].message.content if resp.choices and resp.choices[0].message else ""
    return (content or "").strip()

def generate_html(
    system: str, 
    user: str, 
    model: Optional[str] = None, 
    temperature: float = 0.3
) -> str:
    """
    Call the chat model and return raw HTML.
    """
    client = _get_client()
    model_name = model or os.getenv("MODEL_NAME", "gpt-4o-mini")

    # Add HTML-specific instruction to system prompt
    html_system = system + "\n\nIMPORTANT: Output ONLY raw HTML. Use semantic HTML tags: <h1>, <h2>, <h3>, <p>, <a>, <ul>, <ol>, <li>, <strong>, <em>. NO markdown syntax. NO code fences."

    try:
        resp = client.chat.completions.create(
            model=model_name,
            temperature=temperature,
            messages=[
                {"role": "system", "content": html_system},
                {"role": "user",   "content": user},
            ],
        )
    except (APIConnectionError, RateLimitError, APIError) as e:
        raise RuntimeError(f"OpenAI API error: {e}") from e

    content = resp.choices[0].message.content if resp.choices and resp.choices[0].message else ""
    html = (content or "").strip()
    
    # Clean up any markdown artifacts that might slip through
    html = html.replace("```html", "").replace("```", "").strip()
    
    return html