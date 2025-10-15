# src/offers_layer.py
from __future__ import annotations

import os
import re
import json
from functools import lru_cache
from typing import Any, Dict, List, Optional

import pandas as pd

# --- Streamlit (optional; used only for cache decoration & messages) ---
try:
    import streamlit as st  # type: ignore
    _HAS_ST = True
except Exception:
    _HAS_ST = False

# --- Google Sheets API (service account) ---
from google.oauth2 import service_account
from googleapiclient.discovery import build

# =========================
# Env / Config
# =========================
SHEET_ID = os.getenv("OFFERS_SHEET_ID", "").strip()
TAB_NAME = (os.getenv("OFFERS_SHEET_TAB") or os.getenv("OFFERS_WORKSHEET") or "").strip()

CSV_FALLBACK = os.getenv("OFFERS_CSV_PATH", "data/offers.csv").strip()

SA_PATH = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
SA_TEXT = os.getenv("GOOGLE_SERVICE_ACCOUNT_TEXT", "").strip()

# =========================
# Utilities
# =========================
def _a1(tab: str, c1: str, r1: int, c2: str, r2: int) -> str:
    """Build valid A1 range; quote sheet if it has spaces/symbols."""
    needs_quotes = any(ch in tab for ch in " +-&()!./")
    sheet = f"'{tab}'" if needs_quotes else tab
    return f"{sheet}!{c1}{r1}:{c2}{r2}"

# Add to the top of offers_layer.py with other imports
import time

# Then add this function at module level (same indentation as _load_from_sheets)
import pickle
from datetime import datetime, timedelta

CACHE_FILE = "data/offers_cache.pkl"
CACHE_DURATION = timedelta(hours=6)

def _load_from_dynamic_sheet() -> pd.DataFrame:
    """Load offers dynamically with 6-hour cache."""
    # Check cache first
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'rb') as f:
                cached_data = pickle.load(f)
                cache_age = datetime.now() - cached_data['timestamp']
                
                if cache_age < CACHE_DURATION:
                    if _HAS_ST:
                        st.info(f"✓ Using cached offers (loaded {cached_data['timestamp'].strftime('%I:%M %p')}, expires in {(CACHE_DURATION - cache_age).seconds // 60} min)")
                    return cached_data['df']
        except Exception as e:
            if _HAS_ST:
                st.warning(f"Cache read failed: {e}")
    
    # Cache miss or expired - load from sheets
    if not SHEET_ID:
        raise RuntimeError("OFFERS_SHEET_ID is not set.")
    
    if not TAB_NAME:
        raise RuntimeError("OFFERS_SHEET_TAB is not set.")
    
    # Build sheets service with WRITE permissions
    scopes = ["https://www.googleapis.com/auth/spreadsheets"]
    if SA_TEXT:
        info = json.loads(SA_TEXT)
        creds = service_account.Credentials.from_service_account_info(info, scopes=scopes)
    elif SA_PATH and os.path.exists(SA_PATH):
        creds = service_account.Credentials.from_service_account_file(SA_PATH, scopes=scopes)
    elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        creds = service_account.Credentials.from_service_account_file(
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS"), scopes=scopes
        )
    else:
        raise RuntimeError("Google credentials not found.")
    
    service = build("sheets", "v4", credentials=creds, cache_discovery=False)
    sheet = service.spreadsheets()
    
    # Get all offers
    offers_source_range = "'Offers + Codes - US'!A3:A1600"
    resp = sheet.values().get(
        spreadsheetId=SHEET_ID,
        range=offers_source_range,
        valueRenderOption="UNFORMATTED_VALUE"
    ).execute()
    
    offer_names = [row[0] for row in resp.get("values", []) if row and row[0]]
    
    if not offer_names:
        raise RuntimeError("No offers found")
    
    if _HAS_ST:
        st.info(f"Loading {len(offer_names)} offers from sheet (this takes ~{len(offer_names) * 0.5:.0f}s)...")
    
    offers_data = []
    page_type = "Top Stories Articles"
    shortcode_type = "Promo Card"
    
    progress_bar = None
    if _HAS_ST:
        progress_bar = st.progress(0)
    
    for idx, offer_name in enumerate(offer_names):
        try:
            # Update cells
            updates = [
                {"range": f"{TAB_NAME}!B2", "values": [[offer_name]]},
                {"range": f"{TAB_NAME}!G2", "values": [[page_type]]},
                {"range": f"{TAB_NAME}!I2", "values": [[shortcode_type]]}
            ]
            
            sheet.values().batchUpdate(
                spreadsheetId=SHEET_ID,
                body={"data": updates, "valueInputOption": "USER_ENTERED"}
            ).execute()
            
            time.sleep(0.5)
            
            # Read computed values
            read_range = f"{TAB_NAME}!A2:K2"
            result = sheet.values().get(
                spreadsheetId=SHEET_ID,
                range=read_range,
                valueRenderOption="UNFORMATTED_VALUE"
            ).execute()
            
            row_values = result.get("values", [[]])[0]
            
            if len(row_values) >= 6:
                offers_data.append({
                    "property": row_values[0] if len(row_values) > 0 else "",
                    "affiliate_offer": row_values[1] if len(row_values) > 1 else "",
                    "offer_text": row_values[2] if len(row_values) > 2 else "",
                    "states": row_values[3] if len(row_values) > 3 else "",
                    "terms": row_values[4] if len(row_values) > 4 else "",
                    "bonus_code": row_values[5] if len(row_values) > 5 else "",
                    "page_type": row_values[6] if len(row_values) > 6 else "",
                    "context": row_values[7] if len(row_values) > 7 else "",
                    "shortcode_type": row_values[8] if len(row_values) > 8 else "",
                    "shortcode": row_values[9] if len(row_values) > 9 else "",
                    "switchboard_link": row_values[10] if len(row_values) > 10 else "",  # Column K

                })
            
            if progress_bar and (idx + 1) % 5 == 0:
                progress_bar.progress((idx + 1) / len(offer_names))
        
        except Exception as e:
            if _HAS_ST:
                st.warning(f"Failed: '{offer_name}': {e}")
            continue
    
    if progress_bar:
        progress_bar.progress(1.0)
        progress_bar.empty()
    
    if not offers_data:
        raise RuntimeError("No offers loaded")
    
    df = pd.DataFrame(offers_data)
    df = _normalize_offers_df(df)
    
    # Cache the results
    os.makedirs("data", exist_ok=True)
    with open(CACHE_FILE, 'wb') as f:
        pickle.dump({'df': df, 'timestamp': datetime.now()}, f)
    
    if _HAS_ST:
        st.success(f"✓ Loaded {len(df)} offers and cached for {CACHE_DURATION.seconds // 3600}h")
    
    return df

def _normalize_token(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (s or "").lower())

def _build_sheets_service():
    if not SHEET_ID:
        raise RuntimeError("OFFERS_SHEET_ID is not set.")
    scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
    if SA_TEXT:
        info = json.loads(SA_TEXT)
        creds = service_account.Credentials.from_service_account_info(info, scopes=scopes)
    elif SA_PATH and os.path.exists(SA_PATH):
        creds = service_account.Credentials.from_service_account_file(SA_PATH, scopes=scopes)
    elif os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        creds = service_account.Credentials.from_service_account_file(
            os.getenv("GOOGLE_APPLICATION_CREDENTIALS"), scopes=scopes
        )
    else:
        raise RuntimeError(
            "Google credentials not found. Set GOOGLE_SERVICE_ACCOUNT_JSON or GOOGLE_SERVICE_ACCOUNT_TEXT."
        )
    return build("sheets", "v4", credentials=creds, cache_discovery=False).spreadsheets()

def _read_tab_as_table(sheet, tab: str) -> pd.DataFrame:
    """Read the tab as a simple table. First row = header."""
    rng = _a1(tab, "A", 1, "K", 2000)
    resp = sheet.values().get(
        spreadsheetId=SHEET_ID,
        range=rng,
        valueRenderOption="UNFORMATTED_VALUE",
    ).execute()
    values = resp.get("values", [])
    if not values:
        return pd.DataFrame()
    header = [str(h).strip() for h in values[0]]
    rows = values[1:]
    for r in rows:
        if len(r) < len(header):
            r += [""] * (len(header) - len(r))
    df = pd.DataFrame(rows, columns=header)
    df = df.replace("", pd.NA).dropna(how="all")
    return df

def _map_header(col: str) -> str:
    """Map many possible header variants -> normalized keys."""
    t = _normalize_token(col)
    if t.startswith("property"):
        return "property"
    if t in ("affiliateoffer", "affiliate", "offer", "affiliateofferbasedonthepagetypeselected") or "affiliateoffer" in t:
        return "affiliate_offer"
    if t.startswith("offertext") or "offernarrative" in t:
        return "offer_text"
    if t in ("states", "statelist"):
        return "states"
    if t.startswith("terms") or "legal" in t or "disclaimer" in t:
        return "terms"
    if t.startswith("bonuscode") or t == "code" or "promocode" in t:
        return "bonus_code"
    if t.startswith("pagetype"):
        return "page_type"
    if t == "context":
        return "context"
    if t.startswith("shortcodetype"):
        return "shortcode_type"
    if t == "shortcode":
        return "shortcode"
    if "switchboardlink" in t or t in ("link", "url"):
        return "switchboard_link"
    return _normalize_token(col)

def _parse_brand(affiliate_offer: str) -> str:
    s = (affiliate_offer or "").strip()
    return s.split(":", 1)[0].strip() if ":" in s else s

def _parse_states(states: Any) -> List[str]:
    if not isinstance(states, str) or not states.strip():
        return []
    txt = states.strip()
    if txt.upper() == "ALL":
        return ["ALL"]
    parts = [p.strip().upper() for p in re.split(r"[,\|/]+", txt) if p.strip()]
    out, seen = [], set()
    for p in parts:
        if p not in seen:
            seen.add(p); out.append(p)
    return out

def _normalize_offers_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.rename(columns={c: _map_header(c) for c in df.columns})

    for c in ("property","affiliate_offer","offer_text","states","terms",
              "bonus_code","page_type","context","shortcode_type","shortcode","switchboard_link"):
        if c not in df.columns:
            df[c] = ""

    # derived
    df["brand"] = df["affiliate_offer"].map(_parse_brand)
    df["states_list"] = df["states"].map(_parse_states)
    df["brand_clean"] = df["brand"].astype(str).str.strip()
    df["page_type_clean"] = df["page_type"].astype(str).str.strip()

    df["offer_id"] = (
        df["brand_clean"].astype(str)
        + " | " + df["affiliate_offer"].astype(str).str.strip()
        + " | " + df["bonus_code"].astype(str).str.strip()
    )

    for c in ["offer_text", "terms", "switchboard_link", "shortcode"]:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)

    df = df[df["affiliate_offer"].astype(str).str.strip().ne("")].copy()
    df = df.drop_duplicates(subset=["offer_id"]).reset_index(drop=True)
    return df

# =========================
# Loaders (Sheets → DF with CSV fallback)
# =========================
def _load_from_sheets() -> pd.DataFrame:
    if not TAB_NAME:
        raise RuntimeError("OFFERS_SHEET_TAB / OFFERS_WORKSHEET not set.")
    return _load_from_dynamic_sheet()

def _load_offers_df_uncached() -> pd.DataFrame:
    try:
        return _load_from_sheets()
    except Exception as e:
        if os.path.exists(CSV_FALLBACK):
            df = pd.read_csv(CSV_FALLBACK)
            df = _normalize_offers_df(df)
            if _HAS_ST:
                st.warning(f"Offers: using CSV fallback ({CSV_FALLBACK}). Reason: {e}")
            return df
        raise

# Public cached accessor
if _HAS_ST:
    @st.cache_data(show_spinner=False)
    def get_offers_df_cached() -> pd.DataFrame:
        return _load_offers_df_uncached()
else:
    @lru_cache(maxsize=1)
    def get_offers_df_cached() -> pd.DataFrame:  # type: ignore[no-redef]
        return _load_offers_df_uncached()

def refresh_offers_cache() -> None:
    """Clear cache safely; used by the UI refresh button."""
    try:
        get_offers_df_cached.clear()  # Streamlit cache
    except Exception:
        pass
    try:
        get_offers_df_cached.cache_clear()  # lru_cache fallback
    except Exception:
        pass

# ----------------- Back-compat wrapper your app calls -----------------
def fetch_dynamic_offers(sheet_id: str | None = None,
                         worksheet: str | None = None,
                         sample_cell: str | None = None) -> pd.DataFrame:
    global SHEET_ID, TAB_NAME
    old_sheet, old_tab = SHEET_ID, TAB_NAME
    try:
        if sheet_id:  SHEET_ID = sheet_id.strip()
        if worksheet: TAB_NAME = worksheet.strip()
        refresh_offers_cache()
        return get_offers_df_cached()
    finally:
        SHEET_ID, TAB_NAME = old_sheet, old_tab

# =========================
# Selection helpers used by app.py
# =========================
def list_brands(df: pd.DataFrame) -> List[str]:
    if df.empty or "brand" not in df.columns:
        return []
    vals = sorted({str(b).strip() for b in df["brand"] if str(b).strip()})
    return vals or ["(All Brands)"]

def list_offers_for_brand(df: pd.DataFrame, brand: str) -> List[Dict[str, str]]:
    """Return list of {'id','label'} filtered by brand (or all if blank)."""
    if df.empty:
        return []
    dfx = df
    if brand and brand != "(All Brands)":
        dfx = df[df["brand"].astype(str).str.strip().str.lower() == brand.strip().lower()]
    out: List[Dict[str, str]] = []
    for _, r in dfx.iterrows():
        main = (str(r.get("affiliate_offer") or r.get("offer_text") or r.get("brand"))).strip()
        code = str(r.get("bonus_code") or "").strip()
        label = f"{main}{f' (code: {code})' if code else ''}"
        out.append({"id": str(r["offer_id"]), "label": label})
    return out

# Alias some older names your code references
offers_for_brand = list_offers_for_brand

def list_all_offers(df: pd.DataFrame) -> List[Dict[str, str]]:
    """Flat list of all offers for single-dropdown UIs."""
    return list_offers_for_brand(df, brand="")

def get_offer_by_id(df: pd.DataFrame, offer_id: str) -> Dict[str, Any]:
    if df.empty:
        return {}
    row = df[df["offer_id"] == offer_id]
    if row.empty:
        return {}
    rec = row.iloc[0].to_dict()
    rec.setdefault("brand", _parse_brand(rec.get("affiliate_offer", "")))
    rec.setdefault("offer_text", "")
    rec.setdefault("bonus_code", "")
    rec.setdefault("switchboard_link", rec.get("url", ""))
    rec.setdefault("states_list", _parse_states(rec.get("states", "")))
    rec.setdefault("terms", "")
    return rec

# Keep the old alias alive
offer_by_id = get_offer_by_id

# =========================
# UI helpers some versions of app.py import
# =========================
def load_offers_dynamic() -> pd.DataFrame:
    """Convenience alias used in some earlier app.py versions."""
    return get_offers_df_cached()

def load_offers_csv(path: str = CSV_FALLBACK) -> pd.DataFrame:
    df = pd.read_csv(path)
    return _normalize_offers_df(df)

def default_title_for_offer(offer_row: Dict[str, Any]) -> str:
    brand = (offer_row.get("brand") or "").strip()
    main  = (offer_row.get("offer_text") or offer_row.get("affiliate_offer") or "").strip()
    if brand and main:
        return f"{brand} Promo: {main}"
    return main or brand or "Sportsbook Promo"

# =========================
# CTA markdown renderer (used by prompt_factory)
# =========================
def render_offer_block(offer_row: Dict[str, Any], placement: str = "inline") -> str:
    def _g(k, alt=None):
        v = offer_row.get(k)
        if (not v) and alt:
            v = offer_row.get(alt, "")
        return (v or "").strip()

    brand = _g("brand")
    headline = _g("offer_text") or _g("affiliate_offer")
    code = _g("bonus_code")
    url = _g("switchboard_link") or _g("url") or "#"
    terms = _g("terms")

    title = f"**{brand} Promo**" if brand else "**Promo**"
    code_line = f"\n**Bonus code:** `{code}`" if code else ""
    link_line = f"\n[Claim Offer]({url})" if url and url != "#" else ""
    block = f"> {title}  \n> {headline}  \n{code_line}{link_line}\n"
    if terms:
        block += f"\n<details><summary>Terms apply</summary><p>{terms}</p></details>\n"
    block += "\n21+. Gambling problem? Call 1-800-GAMBLER. Please bet responsibly.\n"
    block = re.sub(r"\n{3,}", "\n\n", block).strip() + "\n"
    return block

