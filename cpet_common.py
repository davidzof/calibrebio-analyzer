# -*- coding: utf-8 -*-
#  © David George, 2025
import re
import pandas as pd

REQUIRED = [
    "Timer (s)",
    "HR (bpm)",
    "Minute Volume (l/min)",
    "VO2 (slpm)",
    "VCO2 (slpm)",
    "VeqO2",
    "VeqCO2",
    "RER",
]

# Canonical plotting names -> exact CSV headers
COLS = {
    "HR": "HR (bpm)",
    "VE": "Minute Volume (l/min)",
    "VO2": "VO2 (slpm)",
    "VCO2": "VCO2 (slpm)",
    "VeqO2": "VeqO2",
    "VeqCO2": "VeqCO2",
    "RER": "RER",
    "Power": "Power",
}

# Flexible aliases -> canonical names
ALIASES = {
    "hr": "HR",
    "hr (bpm)": "HR",
    "ve": "VE",
    "minute volume": "VE",
    "minute volume (l/min)": "VE",
    "vo2": "VO2",
    "vo2 (slpm)": "VO2",
    "vco2": "VCO2",
    "vco2 (slpm)": "VCO2",
    "veqo2": "VeqO2",
    "veq o2": "VeqO2",
    "veqco2": "VeqCO2",
    "veq co2": "VeqCO2",
    "rer": "RER",
    "power": "Power",
    "watts": "Power",
    "w": "Power",
    "pwr": "Power",
    "cycling power (w)": "Power",  # legacy header accepted on CLI
    "cycling power": "Power",
}

def norm_alias(s: str) -> str:
    import re as _re
    return _re.sub(r"\s+", " ", s.strip().lower())

def to_canonical(keys):
    if not keys:
        return []
    out = []
    for k in keys:
        nk = norm_alias(k)
        canon = ALIASES.get(nk)
        if canon and canon in COLS:
            out.append(canon)
    # deduplicate preserving order
    seen, dedup = set(), []
    for x in out:
        if x not in seen:
            seen.add(x); dedup.append(x)
    return dedup

def seconds_to_rows(time_s: pd.Series, seconds: float) -> int:
    if not seconds or seconds <= 0:
        return 0
    t = pd.to_numeric(time_s, errors="coerce")
    dt = t.diff().median()
    if pd.isna(dt) or dt <= 0:
        dt = 1.0
    return max(3, int(round(seconds / float(dt))))

def smooth_series(s: pd.Series, win_rows: int) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if not win_rows or win_rows <= 1:
        return x
    return x.rolling(window=win_rows, center=True, min_periods=max(1, win_rows//3)).median()

def parse_trim(df: pd.DataFrame, raw: str) -> pd.DataFrame:
    raw = raw.strip()
    start_s, end_s = None, None
    if "," in raw:
        a, b = raw.split(",", 1)
        a = a.strip(); b = b.strip()
        start_s = float(a) if a else None
        end_s = float(b) if b else None
    else:
        start_s = float(raw)
    t_all = pd.to_numeric(df["Timer (s)"], errors="coerce")
    mask = pd.Series(True, index=df.index)
    if start_s is not None:
        mask &= t_all >= start_s
    if end_s is not None:
        mask &= t_all <= end_s
    return df.loc[mask].reset_index(drop=True)

def validate_required(df: pd.DataFrame):
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required column(s): {missing}")
