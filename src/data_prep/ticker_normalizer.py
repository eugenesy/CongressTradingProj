"""
ticker_normalizer.py
====================
Conservative ticker normalizer for the congressional trading dataset.

Design principles:
  - NEVER match a ticker to a parquet file unless we are confident it is
    the correct company.  It is better to drop a row than link wrong prices.
  - Every normalization step is explicit and documented.
  - A full unmatched report is printed at the end so we know exactly what
    was dropped and why.

Usage:
    from scripts.build_v2.ticker_normalizer import TickerNormalizer

    normalizer = TickerNormalizer(parquet_dir="data/parquet")
    normalizer.fit()                          # scan parquet files once
    df["Matched_Ticker"] = normalizer.transform(df["Ticker"])
    normalizer.print_report(df["Ticker"])
"""

import os
import re
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hard-coded ticker corrections
# These are cases where automated normalization would be ambiguous or wrong.
# Format: {raw_ticker: correct_parquet_stem} or {raw_ticker: None} to drop.
# ---------------------------------------------------------------------------
MANUAL_MAP: dict[str, Optional[str]] = {
    # Berkshire Hathaway — BRK.parquet covers both classes in source data
    "BRK.B":  "BRK",
    "BRK.A":  "BRK",
    "BRK-A":  "BRK",
    "BRK-B":  "BRK",
    # Lion's Gate Entertainment share classes
    "LGF.A":  "LGF",
    "LGF.B":  "LGF",
    "LGF-A":  "LGF",
    "LGF-B":  "LGF",
    # Lannett Company (delisted) – no parquet
    "LCI":    None,
    # Royal Dutch Shell (pre-merger naming)
    "RDS.A":  "RDSMY",
    "RDS.B":  "RDSMY",
    "RDS-A":  "RDSMY",
    "RDS-B":  "RDSMY",
    # Lennar Corp B shares
    "LEN.B":  "LEN",
    "LEN-B":  "LEN",
    # Brown-Forman
    "BF.A":   "BF",
    "BF.B":   "BF",
    "BF-A":   "BF",
    "BF-B":   "BF",
    # Clearway Energy
    "CWEN.A": "CWEN",
    # HEICO Corporation
    "HEI.A":  "HEI",
    # NexGen Energy Ltd warrants — not a traded stock, drop
    "LLY-WD": None,
    # Clearly non-equities — drop
    "LENB_failed": None,
}

# ---------------------------------------------------------------------------
# Patterns that clearly identify non-equity entries — drop these rows.
# ---------------------------------------------------------------------------
NON_EQUITY_PATTERNS = [
    re.compile(r","),                          # "GLAS FUNDS, LP", multi-ticker
    re.compile(r"\s{2,}"),                     # multiple spaces
    re.compile(r"DUE\s+\d"),                   # bond maturity descriptions
    re.compile(r"SYMBOL\s*:"),                 # SYMBOL: OGVXX etc
    re.compile(r"PART\s+INTEREST", re.I),
    re.compile(r"TREASURY", re.I),
    re.compile(r"ONE\s+SHARE", re.I),
    re.compile(r"MATURE", re.I),               # T-bill maturities
    re.compile(r"D/B/A", re.I),
    re.compile(r"\d{15,}"),                    # OCC options: SPY160219P00180000
    re.compile(r"^\d+$"),                      # purely numeric (ticker "2")
    re.compile(r"\.\d+$"),                     # foreign exchange: 1AMT.MI, LM09.SG
    re.compile(r"\.[A-Z]{2}$"),               # exchange suffix: .MI, .SG, .L, .TO
    re.compile(r"NORTHWESTER", re.I),
]

# Ticker types that are not plain equities — drop these TickerType codes.
NON_EQUITY_TICKER_TYPES = {
    "OP", "Stock Option", "Corporate Bond", "Cryptocurrency",
    "OI", "OL", "SA",
}


class TickerNormalizer:
    """
    Matches raw ticker strings from the congressional trading CSV to
    parquet file stems in a given directory.

    Steps tried in order (stops at first successful match):
      1. Direct match: "AAPL" -> "AAPL.parquet"
      2. Manual map: hard-coded corrections above
      3. Strip exchange prefix: "NASDAQ:WLTW" -> "WLTW"
      4. Strip _failed suffix: "LENB_failed" -> "LENB"
      5. Numeric suffix strip: "WB1" -> "WB" (only if result is in parquets)
      6. Dot-class to base: "BRK.B" -> "BRK" (only if result is in parquets)
      7. Dash-class to base: "BRK-A" -> "BRK" (only if result is in parquets)

    If none match, returns None (row will be dropped with a reason logged).
    """

    def __init__(self, parquet_dir: str = "data/parquet"):
        self.parquet_dir = Path(parquet_dir)
        self.available: set[str] = set()      # parquet stems present on disk
        self._cache: dict[str, Optional[str]] = {}
        self._reason: dict[str, str] = {}     # raw_ticker -> drop reason

    def fit(self) -> "TickerNormalizer":
        """Scan the parquet directory and index available tickers."""
        if not self.parquet_dir.exists():
            raise FileNotFoundError(f"Parquet directory not found: {self.parquet_dir}")
        self.available = {
            p.stem for p in self.parquet_dir.glob("*.parquet")
        }
        logger.info("TickerNormalizer: found %d parquet files in %s",
                    len(self.available), self.parquet_dir)
        return self

    def _normalize_one(self, raw: str) -> Optional[str]:
        """Return the matched parquet stem, or None if no safe match exists."""
        if not isinstance(raw, str):
            return None

        ticker = raw.strip().upper()

        # --- Non-equity pattern check (drop before any matching) ---
        for pat in NON_EQUITY_PATTERNS:
            if pat.search(ticker):
                self._reason[raw] = f"non-equity pattern: {pat.pattern!r}"
                return None

        # Step 1: direct match
        if ticker in self.available:
            return ticker

        # Step 2: manual map
        if ticker in MANUAL_MAP:
            mapped = MANUAL_MAP[ticker]
            if mapped is None:
                self._reason[raw] = "manual map: explicitly dropped"
                return None
            if mapped in self.available:
                return mapped
            # Manual map target not on disk — log and drop rather than guess
            self._reason[raw] = f"manual map target {mapped!r} not found on disk"
            return None

        # Step 3: strip exchange prefix (NASDAQ:WLTW -> WLTW)
        if ":" in ticker:
            candidate = ticker.split(":")[-1].strip()
            if candidate in self.available:
                return candidate
            self._reason[raw] = f"stripped-prefix candidate {candidate!r} not found"
            return None

        # Step 4: strip _failed suffix
        if ticker.endswith("_FAILED"):
            candidate = ticker[: -len("_FAILED")]
            if candidate in self.available:
                return candidate
            self._reason[raw] = f"_failed-stripped candidate {candidate!r} not found"
            return None

        # Step 5: numeric suffix (delisted/successor tickers, e.g. WB1 -> WB)
        # Only accept if the stripped version is >=2 chars (avoid single-char accidents)
        num_stripped = re.sub(r"\d+$", "", ticker)
        if num_stripped and num_stripped != ticker and len(num_stripped) >= 2:
            if num_stripped in self.available:
                return num_stripped
            # Do NOT fall through: numeric suffix tickers are usually delisted
            # companies whose successor may have a totally different ticker.
            self._reason[raw] = f"numeric-suffix ticker; stripped {num_stripped!r} not found — likely delisted"
            return None

        # Step 6: dot-class notation (BRK.B -> BRK) — only base class
        if "." in ticker:
            base = ticker.split(".")[0]
            if len(base) >= 1 and base in self.available:
                return base
            self._reason[raw] = f"dot-class base {base!r} not found in parquets"
            return None

        # Step 7: dash-class notation (BRK-A -> BRK, LEN-B -> LEN)
        # Only strip a single suffix component (one dash)
        if "-" in ticker:
            base = ticker.split("-")[0]
            if len(base) >= 2 and base in self.available:
                return base
            # Multi-dash (SHO-P-I) or base not found — drop
            self._reason[raw] = f"dash-class base {base!r} not found in parquets"
            return None

        # No match found
        self._reason[raw] = "no matching parquet file found"
        return None

    def transform(self, ticker_series: pd.Series) -> pd.Series:
        """
        Map a Series of raw ticker strings to matched parquet stems.
        Returns a Series of the same index; unmatched rows contain None/NaN.
        """
        result = ticker_series.map(self._normalize_cached)
        return result

    def _normalize_cached(self, raw: str) -> Optional[str]:
        if raw not in self._cache:
            self._cache[raw] = self._normalize_one(raw)
        return self._cache[raw]

    def print_report(self, ticker_series: pd.Series) -> None:
        """Print a summary of unmatched tickers with counts and reasons."""
        matched = ticker_series.map(self._normalize_cached)
        total = len(ticker_series)
        n_matched = matched.notna().sum()
        n_dropped = matched.isna().sum()

        print(f"\n{'='*60}")
        print(f"TICKER NORMALIZATION REPORT")
        print(f"{'='*60}")
        print(f"Total rows        : {total:>8,}")
        print(f"Matched           : {n_matched:>8,}  ({n_matched/total*100:.1f}%)")
        print(f"Dropped (no match): {n_dropped:>8,}  ({n_dropped/total*100:.1f}%)")
        print()

        if n_dropped > 0:
            print("Unmatched tickers (by row count):")
            unmatched_mask = matched.isna()
            counts = ticker_series[unmatched_mask].value_counts()
            for raw_tick, cnt in counts.items():
                reason = self._reason.get(raw_tick.strip().upper(),
                                          self._reason.get(raw_tick, "unknown"))
                print(f"  {raw_tick:30s}  {cnt:5d} rows  [{reason}]")
        print(f"{'='*60}\n")

    def get_unmatched(self, ticker_series: pd.Series) -> pd.DataFrame:
        """Return a DataFrame of unmatched tickers with counts and reasons."""
        matched = ticker_series.map(self._normalize_cached)
        unmatched_mask = matched.isna()
        if not unmatched_mask.any():
            return pd.DataFrame(columns=["Ticker", "Count", "Reason"])
        counts = ticker_series[unmatched_mask].value_counts().reset_index()
        counts.columns = ["Ticker", "Count"]
        counts["Reason"] = counts["Ticker"].map(
            lambda t: self._reason.get(t.strip().upper(),
                                       self._reason.get(t, "unknown"))
        )
        return counts
