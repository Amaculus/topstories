import os, re, csv
from pathlib import Path
from typing import Dict, List, Tuple

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

SPREADSHEET_ID = os.getenv("OFFERS_SHEET_ID")  # 1pXgc3fY...
WORKSHEET_TITLE = os.getenv("OFFERS_WORKSHEET", "Ultimate Builder")
SERVICE_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE")  # C:\...\service.json
SAMPLE_CELL = os.getenv("OFFERS_SAMPLE_CELL", "B2")  # cell with Affiliate/Offer dropdown
OUT_CSV = Path("data/offers_extracted.csv")
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

def _creds():
    if not SERVICE_FILE or not Path(SERVICE_FILE).exists():
        raise SystemExit("GOOGLE_SERVICE_ACCOUNT_FILE not set or not found.")
    return Credentials.from_service_account_file(SERVICE_FILE, scopes=SCOPES)

def _col_to_a1(idx: int) -> str:
    s = ""
    c = idx + 1
    while c:
        c, rem = divmod(c - 1, 26)
        s = chr(65 + rem) + s
    return s

def _a1_from_gridrange(grid_range: Dict, sheet_title: str) -> str:
    sr = grid_range.get("startRowIndex", 0)
    er = grid_range.get("endRowIndex", sr + 1)
    sc = grid_range.get("startColumnIndex", 0)
    ec = grid_range.get("endColumnIndex", sc + 1)
    return f"'{sheet_title}'!{_col_to_a1(sc)}{sr+1}:{_col_to_a1(ec-1)}{er}"

def _parse_validation_ref(spreadsheet_meta: Dict, dv: Dict) -> Tuple[str, str]:
    cond = (dv or {}).get("condition", {})
    vals = cond.get("values", [])
    if not vals:
        raise ValueError("No values in data validation.")
    raw = vals[0].get("userEnteredValue", "")
    ref = raw.lstrip("=")

    # Named range?
    if "!" not in ref:
        for nr in spreadsheet_meta.get("namedRanges", []):
            if nr.get("name") == ref:
                sid = nr["range"]["sheetId"]
                title = next(s["properties"]["title"] for s in spreadsheet_meta["sheets"]
                             if s["properties"]["sheetId"] == sid)
                return title, _a1_from_gridrange(nr["range"], title)
        raise ValueError(f"Could not resolve named range: {ref}")

    # Sheet!Range
    m = re.match(r"^'?([^'!]+)'?!([$A-Z0-9:]+)$", ref)
    if not m:
        raise ValueError(f"Unrecognized validation ref: {ref}")
    return m.group(1), f"'{m.group(1)}'!{m.group(2)}"

def _values(service, rng: str) -> List[List[str]]:
    return service.spreadsheets().values().get(
        spreadsheetId=SPREADSHEET_ID, range=rng).execute().get("values", [])

def _header_map(headers: List[str]) -> Dict[str, int]:
    return { (h or "").strip().lower(): i for i, h in enumerate(headers) }

def _expand_to_table(src_rng: str, headers: List[str]) -> str:
    # `'Offers'!B2:B1000` -> expand width to all header columns on same rows
    m = re.match(r"^'([^']+)'!\$?([A-Z]+)\$?(\d+):\$?([A-Z]+)\$?(\d+)$", src_rng)
    if not m:
        return src_rng
    sheet, _, r1, _, r2 = m.groups()
    last = _col_to_a1(len(headers) - 1)
    return f"'{sheet}'!A{r1}:{last}{r2}"

def _brand_from_affiliate(affiliate_offer: str) -> str:
    # "DraftKings National Offer: ..." -> "DraftKings"
    if not affiliate_offer:
        return ""
    return affiliate_offer.split(":")[0].split("—")[0].split("-")[0].strip()

def main():
    if not SPREADSHEET_ID:
        raise SystemExit("OFFERS_SHEET_ID missing")

    service = build("sheets", "v4", credentials=_creds())

    # Pull validation on B2
    resp = service.spreadsheets().get(
        spreadsheetId=SPREADSHEET_ID,
        includeGridData=True,
        ranges=[f"'{WORKSHEET_TITLE}'!{SAMPLE_CELL}"],
    ).execute()
    if not resp.get("sheets"):
        raise SystemExit("Worksheet/cell not found")

    # Need full meta for namedRanges
    meta = service.spreadsheets().get(spreadsheetId=SPREADSHEET_ID).execute()

    dv = (resp["sheets"][0]["data"][0]["rowData"][0]["values"][0]).get("dataValidation")
    if not dv:
        raise SystemExit(f"No data validation on {WORKSHEET_TITLE}!{SAMPLE_CELL}")

    src_sheet, src_rng = _parse_validation_ref(meta, dv)

    # headers from first row of the source sheet
    headers = _values(service, f"'{src_sheet}'!1:1")
    headers = headers[0] if headers else []
    hmap = _header_map(headers)
    table_rng = _expand_to_table(src_rng, headers)
    rows = _values(service, table_rng)

    # Build output
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_rows = []
    for r in rows:
        # Basic required field
        aff_idx = hmap.get("affiliate/offer") or hmap.get("affiliate") or hmap.get("offer")
        if aff_idx is None or aff_idx >= len(r) or not (r[aff_idx] or "").strip():
            continue

        rec = {
            "affiliate_offer": r[aff_idx].strip(),
            "brand": _brand_from_affiliate(r[aff_idx]),
        }

        for key in ("offer text", "states", "terms", "bonus code", "switchboard link", "url"):
            i = hmap.get(key)
            if i is not None and i < len(r):
                rec[key.replace(" ", "_")] = r[i]

        out_rows.append(rec)

    # Write CSV
    fieldnames = sorted({k for row in out_rows for k in row.keys()})
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(f"Source: {src_sheet} via {src_rng}")
    print(f"Offers exported: {len(out_rows)} → {OUT_CSV.resolve()}")
    for rec in out_rows[:5]:
        print(" •", rec["affiliate_offer"], "→", rec.get("offer_text", "")[:60])

if __name__ == "__main__":
    main()
