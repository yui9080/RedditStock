#!/usr/bin/env python3
"""
Build an approximate public-data substitute for the CRSP SHRCD 10/11 ticker universe.

IMPORTANT:
- This is NOT CRSP and does not reproduce SHRCD exactly.
- It is intended for the text-matching / feasibility stage of a replication.
- The output deliberately keeps a reasonably broad set of likely common equities.
- For a final paper replication, replace this with CRSP stocknames + SHRCD in {10,11}.

Source:
https://github.com/asad70/us_symbol_list
Files:
  nasdaqsym.csv
  nysesym.csv
  amexsym.csv
"""

from __future__ import annotations

import csv
import io
import re
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/asad70/us_symbol_list/main/"
FILES = {
    "NASDAQ": "nasdaqsym.csv",
    "NYSE": "nysesym.csv",
    "AMEX": "amexsym.csv",
}

from pipeline_io import new_run, publish
ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = new_run(ROOT, 'universe')
OUT = RUN_DIR/'approx_common_stock_universe.csv'

# Strong indicators that the listed security itself is not ordinary common equity.
# These are intentionally conservative: better to keep an ambiguous operating company
# for initial Reddit matching than to delete a true common stock by mistake.
EXCLUDE_PATTERNS = [
    r"\bETF\b",
    r"EXCHANGE[- ]TRADED FUND",
    r"EXCHANGE[- ]TRADED NOTE",
    r"\bETN\b",
    r"\bWARRANTS?\b",
    r"\bWTS?\b(?:\s|$)",
    r"\bRIGHTS?\b",
    r"\bUNITS?\b",
    r"\bPREFERRED\b",
    r"\bPFD\b",
    r"\bPREFERENCE SHARES?\b",
    r"\bDEPOSITARY SHARES?\b.*PREFERRED",
    r"\bCLOSED[- ]END\b",
    r"\bMUTUAL FUND\b",
    r"\bBOND FUND\b",
    r"\bINDEX FUND\b",
    r"\bTRUST PREFERRED\b",
    r"\bNOTES?\b.*(?:DUE|MATUR)",
]

# CRSP SHRCD 10/11 excludes ADR/ADS-type foreign receipts.
ADR_PATTERNS = [
    r"\bADR\b",
    r"\bADS\b",
    r"AMERICAN DEPOSITARY",
    r"DEPOSITARY RECEIPT",
    r"DEPOSITORY RECEIPT",
]

# Name patterns that are very likely a REIT/security rather than ordinary industrial common equity.
# This cannot catch every REIT; flagging is retained in the output for manual review.
REIT_PATTERNS = [
    r"\bREIT\b",
    r"REAL ESTATE INVESTMENT TRUST",
]

# Ticker suffixes frequently used for non-common securities.
# Do NOT blindly exclude any ticker ending in W/U/R/P because normal companies can end in these letters.
# We only use these as a secondary flag when the security name is also suspicious.
SUSPICIOUS_SUFFIX = re.compile(r"(?:W|WS|WT|U|R|P)$", re.I)

def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "academic-replication/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8-sig", errors="replace")

def matches_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(p, text, flags=re.I) for p in patterns)

def normalize_ticker(x: str) -> str:
    return x.strip().upper()

rows = []
for exchange, fn in FILES.items():
    text = fetch(BASE + fn)
    reader = csv.reader(io.StringIO(text))
    for rec in reader:
        if len(rec) < 2:
            continue
        company = rec[0].strip()
        ticker = normalize_ticker(rec[1])
        if not company or not ticker:
            continue

        # Paper: keep tickers with at least 3 letters.
        # Requiring alphabetic tickers is intentional for Reddit token matching.
        if len(ticker) < 3 or not ticker.isalpha():
            continue

        # Paper excludes GameStop in the main sample.
        if ticker == "GME":
            continue

        upper_name = company.upper()

        is_adr = matches_any(upper_name, ADR_PATTERNS)
        is_reit = matches_any(upper_name, REIT_PATTERNS)
        is_noncommon = matches_any(upper_name, EXCLUDE_PATTERNS)

        # Name + suffix combination catches many SPAC units/warrants/rights.
        suffix_suspicious = bool(SUSPICIOUS_SUFFIX.search(ticker))
        name_suspicious = matches_any(
            upper_name,
            [
                r"\bSPAC\b",
                r"ACQUISITION CORPORATION",
                r"ACQUISITION CORP",
                r"\bWARRANT",
                r"\bUNIT",
                r"\bRIGHT",
                r"\bPREFERRED",
            ],
        )

        keep = not (is_adr or is_reit or is_noncommon or (suffix_suspicious and name_suspicious))

        reasons = []
        if is_adr:
            reasons.append("ADR/ADS")
        if is_reit:
            reasons.append("REIT")
        if is_noncommon:
            reasons.append("non-common security keyword")
        if suffix_suspicious and name_suspicious:
            reasons.append("suspicious ticker suffix + security name")

        rows.append({
            "ticker": ticker,
            "company_name": company,
            "exchange": exchange,
            "delisted_flag": int("(delisted)" in company.lower()),
            "approx_common_stock": int(keep),
            "exclusion_reason": "; ".join(reasons),
            "source_file": fn,
        })

# A ticker can appear in more than one historical record/exchange.
# Keep all records in the detailed output, and also write a deduplicated keep-list.
rows.sort(key=lambda x: (x["ticker"], x["exchange"], x["company_name"]))

with OUT.open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

keep_tickers = sorted({r["ticker"] for r in rows if r["approx_common_stock"] == 1})
with (RUN_DIR/'approx_common_stock_tickers.txt').open("w", encoding="utf-8") as f:
    f.write("\n".join(keep_tickers) + "\n")

print(f"Wrote {OUT} with {len(rows):,} security records.")
print(f"Approximate common-stock ticker dictionary: {len(keep_tickers):,} unique tickers.")
print('Ticker dictionary:', RUN_DIR/'approx_common_stock_tickers.txt')
publish(ROOT,'universe',RUN_DIR,{'source_base':BASE,'source_files':FILES,'security_records':len(rows),'unique_kept_tickers':len(keep_tickers),'dictionary_file':'approx_common_stock_tickers.txt','classification':'approximate public-data heuristic; not CRSP','date_range':None,'unique_authors':None,'unique_threads':None})
