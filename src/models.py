# src/models.py
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class OutlineItem(BaseModel):
    id: str
    level: str  # "h2" or "h3"
    title: str
    must_include: List[str] = []
    optional: List[str] = []
    h3: List["OutlineItem"] = Field(default_factory=list)

class Outline(BaseModel):
    project_id: str
    working_title: str
    angle: str = ""
    structure: List[OutlineItem] = []
    faq_pool: List[str] = []
    internal_link_targets: List[str] = []

class SectionBrief(BaseModel):
    section_id: str
    objective: str
    audience: str
    constraints: Dict[str, Any] = {}
    facts_and_points: List[str] = []
    retrieved_snippets: List[Dict[str, str]] = []

class InlineLinkSpec(BaseModel):
    title: str
    url: str
    recommended_anchors: List[str]

class PromptSect(BaseModel):
    system: str
    user: str
    rules: List[str]
    temperature: float = 0.3  # Default, can be overridden