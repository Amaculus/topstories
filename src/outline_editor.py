# src/outline_editor.py
import streamlit as st, uuid
from .models import Outline, OutlineItem

def _uid(prefix="sec"): 
    return f"{prefix}-{uuid.uuid4().hex[:8]}"

def _ensure_h3_list(item: OutlineItem):
    if item.h3 is None:
        item.h3 = []

def _apply_text_edits(outline: Outline):
    """Pull values from session_state into the Outline object (autosave)."""
    # H2s
    for item in outline.structure:
        item.title = st.session_state.get(f"title_{item.id}", item.title)
        mi = st.session_state.get(f"mi_{item.id}", "")
        op = st.session_state.get(f"op_{item.id}", "")
        item.must_include = [x.strip() for x in mi.split("\n") if x.strip()]
        item.optional    = [x.strip() for x in op.split("\n") if x.strip()]
        _ensure_h3_list(item)
        # H3s
        for sub in item.h3:
            sub.title = st.session_state.get(f"title_{sub.id}", sub.title)
            mi2 = st.session_state.get(f"mi_{sub.id}", "")
            op2 = st.session_state.get(f"op_{sub.id}", "")
            sub.must_include = [x.strip() for x in mi2.split("\n") if x.strip()]
            sub.optional    = [x.strip() for x in op2.split("\n") if x.strip()]

def move_h2(outline: Outline, sec_id: str, delta: int):
    idx = next((i for i,s in enumerate(outline.structure) if s.id==sec_id and s.level=="h2"), None)
    if idx is None: return
    new = max(0, min(len(outline.structure)-1, idx+delta))
    outline.structure[idx], outline.structure[new] = outline.structure[new], outline.structure[idx]

def move_h3(outline: Outline, parent_id: str, h3_id: str, delta: int):
    pidx = next((i for i,s in enumerate(outline.structure) if s.id==parent_id and s.level=="h2"), None)
    if pidx is None: return
    h = outline.structure[pidx].h3
    j = next((i for i,s in enumerate(h) if s.id==h3_id and s.level=="h3"), None)
    if j is None: return
    new = max(0, min(len(h)-1, j+delta))
    h[j], h[new] = h[new], h[j]

def delete_h2(outline: Outline, h2_id: str):
    idx = next((i for i,s in enumerate(outline.structure) if s.id==h2_id and s.level=="h2"), None)
    if idx is not None:
        outline.structure.pop(idx)

def delete_h3(outline: Outline, parent_id: str, h3_id: str):
    pidx = next((i for i,s in enumerate(outline.structure) if s.id==parent_id and s.level=="h2"), None)
    if pidx is None: return
    h = outline.structure[pidx].h3
    j = next((i for i,s in enumerate(h) if s.id==h3_id and s.level=="h3"), None)
    if j is not None:
        h.pop(j)

def add_h2(outline: Outline, title="New Section"):
    outline.structure.append(
        OutlineItem(id=_uid("h2"), level="h2", title=title, must_include=[], optional=[], h3=[])
    )

def add_h3(outline: Outline, parent_id: str, title="New Subsection"):
    pidx = next((i for i,s in enumerate(outline.structure) if s.id==parent_id and s.level=="h2"), None)
    if pidx is None: return
    outline.structure[pidx].h3.append(
        OutlineItem(id=_uid("h3"), level="h3", title=title, must_include=[], optional=[])
    )

def render_editor(outline: Outline):
    st.subheader("Outline Editor (add / edit / move / delete)")
    st.caption("Edits are **auto-saved** as you type. Rearranging/adding/deleting updates instantly.")

    # Add H2
    c1, c2 = st.columns([3,1])
    new_h2_title = c1.text_input("Add H2 title", key="new_h2_title", placeholder="e.g., How to Claim the Offer")
    if c2.button("Add H2"):
        add_h2(outline, title=new_h2_title or "New Section")
        st.rerun()

    # Existing sections
    for item in outline.structure:
        _ensure_h3_list(item)
        with st.expander(f"H2 • {item.title or 'Untitled'}", expanded=False):
            r1, r2, r3, r4 = st.columns([1,1,1,2])
            if r1.button("▲ Move up", key=f"up_{item.id}"):
                move_h2(outline, item.id, -1); st.rerun()
            if r2.button("▼ Move down", key=f"down_{item.id}"):
                move_h2(outline, item.id, +1); st.rerun()
            if r3.button("✚ Add H3", key=f"addh3_{item.id}"):
                add_h3(outline, item.id, title="New Subsection"); st.rerun()
            if r4.button("🗑 Delete H2", key=f"del_{item.id}"):
                delete_h2(outline, item.id); st.rerun()

            st.text_input("H2 Title", value=item.title, key=f"title_{item.id}")
            st.text_area("Must include (one per line)", value="\n".join(item.must_include or []), key=f"mi_{item.id}", height=100)
            st.text_area("Optional (one per line)", value="\n".join(item.optional or []), key=f"op_{item.id}", height=80)

            st.markdown("**H3 Subsections**")
            for sub in list(item.h3):
                s1, s2, s3, s4 = st.columns([1,1,1,2])
                if s1.button("▲", key=f"subup_{sub.id}"):
                    move_h3(outline, item.id, sub.id, -1); st.rerun()
                if s2.button("▼", key=f"subdown_{sub.id}"):
                    move_h3(outline, item.id, sub.id, +1); st.rerun()
                if s4.button("🗑 Delete H3", key=f"subdel_{sub.id}"):
                    delete_h3(outline, item.id, sub.id); st.rerun()

                st.text_input("H3 Title", value=sub.title, key=f"title_{sub.id}")
                st.text_area("Must include (one per line)", value="\n".join(sub.must_include or []), key=f"mi_{sub.id}", height=80)
                st.text_area("Optional (one per line)", value="\n".join(sub.optional or []), key=f"op_{sub.id}", height=60)
                st.divider()

    # AUTOSAVE: pull widget values back into the Outline object every run
    _apply_text_edits(outline)
