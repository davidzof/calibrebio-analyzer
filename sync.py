#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge power from a TCX file into a CPET CSV by aligning the shared heart-rate signal.
 © David George, 2025

Steps
1) Parse CPET CSV -> columns: time_s ("Timer (s)") and HR ("HR (bpm)").
2) Parse TCX -> per-sample time (datetime), HR (bpm), Watts. Make seconds-from-start.
3) Estimate time offset between files by maximizing Pearson r of 1 Hz HR series
   across lags in [-max_shift, +max_shift].
4) Shift TCX by that offset; interpolate Watts onto CPET times.
5) Save merged CSV with QA columns:
   - tcx_hr_interp: TCX HR re-sampled onto CPET times
   - hr_diff_bpm:   CPET_HR - tcx_hr_interp
   - tcx_time_offset_s: applied offset (constant column)
Fallback:
- If correlation-based offset fails, you can force a manual offset via --offset
  or enable --hr-nearest to map each CPET row to the nearest TCX HR within a bpm window.

Usage:
  python sync.py --cpet CPET.csv --tcx ride.tcx --out merged.csv
"""

import argparse
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from typing import Tuple, Optional

import numpy as np
import pandas as pd


# ---------- Parsing ----------

def _parse_tcx_time(s: str) -> datetime:
    # TCX times are ISO8601, often Zulu (UTC)
    # Example: 2025-11-01T10:23:45.000Z
    # datetime.fromisoformat can't handle trailing 'Z' directly pre-3.11, so:
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def read_tcx(tcx_path: str) -> pd.DataFrame:
    """
    Return DataFrame with columns:
      - time (datetime, timezone-aware if provided)
      - hr (bpm)  [float]
      - watts (W) [float or NaN if absent]
      - t_s (seconds from start, float)
    """
    ns = {}  # avoid strict namespaces, match by tag suffix instead
    root = ET.parse(tcx_path).getroot()

    times, hrs, watts = [], [], []

    # Walk through all Trackpoints
    for tp in root.iter():
        if not tp.tag.endswith("Trackpoint"):
            continue

        t_elem = None
        hr_elem = None
        w_elem = None

        for child in tp:
            tag = child.tag.split("}")[-1]
            if tag == "Time":
                t_elem = child.text
            elif tag == "HeartRateBpm":
                # <HeartRateBpm><Value>xxx</Value></HeartRateBpm>
                for sub in child:
                    if sub.tag.split("}")[-1] == "Value":
                        hr_elem = sub.text
            elif tag == "Extensions":
                # Look for TPX/Watts under Extensions (various namespaces)
                for g in child.iter():
                    if g.tag.split("}")[-1] == "Watts":
                        w_elem = g.text

        if t_elem is None:
            continue

        t = _parse_tcx_time(t_elem)
        hr = float(hr_elem) if hr_elem is not None else np.nan
        w = float(w_elem) if w_elem is not None else np.nan

        times.append(t)
        hrs.append(hr)
        watts.append(w)

    if not times:
        raise SystemExit("No Trackpoint samples found in TCX.")

    df = pd.DataFrame({"time": times, "hr": hrs, "watts": watts})
    # Ensure tz-aware (UTC) for math
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize(timezone.utc)
    t0 = df["time"].min()
    df["t_s"] = (df["time"] - t0).dt.total_seconds()
    # Deduplicate and sort
    df = df.sort_values("t_s").drop_duplicates(subset=["t_s"])
    return df.reset_index(drop=True)


def read_cpet(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Be liberal with headers: accept typical CPET export names
    # Required canonical: "Timer (s)" and "HR (bpm)"
    # Try a few variants if needed:
    def pick(cands):
        for c in cands:
            if c in df.columns:
                return c
        # case-insensitive fallback
        low = {c.lower(): c for c in df.columns}
        for c in cands:
            if c.lower() in low:
                return low[c.lower()]
        return None

    t_col = pick(["Timer (s)", "time", "Time (s)", "timer_s", "timer"])
    hr_col = pick(["HR (bpm)", "hr", "Heart Rate", "heart_rate", "HR"])

    if t_col is None or hr_col is None:
        raise SystemExit(f"Could not find time/HR columns in CPET CSV. Columns: {list(df.columns)}")

    out = pd.DataFrame({
        "time_s": pd.to_numeric(df[t_col], errors="coerce"),
        "hr": pd.to_numeric(df[hr_col], errors="coerce")
    })
    out = out.dropna(subset=["time_s"]).sort_values("time_s").reset_index(drop=True)
    return out


# ---------- Alignment ----------

def _series_to_1hz(time_s: np.ndarray, y: np.ndarray, smooth_sec: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    """Resample y(time_s) to 1 Hz on [ceil(min), floor(max)] with linear interp; optional median smoothing."""
    mask = np.isfinite(time_s) & np.isfinite(y)
    time_s = time_s[mask]
    y = y[mask]
    if len(time_s) < 5:
        return np.array([]), np.array([])
    t0 = int(np.ceil(time_s.min()))
    t1 = int(np.floor(time_s.max()))
    if t1 <= t0:
        return np.array([]), np.array([])

    grid = np.arange(t0, t1 + 1, 1, dtype=float)
    yi = np.interp(grid, time_s, y)

    if smooth_sec and smooth_sec > 0:
        w = max(3, int(round(smooth_sec)))
        s = pd.Series(yi).rolling(w, center=True, min_periods=max(1, w // 3)).median()
        yi = s.to_numpy()
    return grid, yi


def estimate_offset_by_hr(cpet: pd.DataFrame,
                          tcx: pd.DataFrame,
                          max_shift: int = 120,
                          smooth_sec: float = 5.0) -> Optional[int]:
    """
    Cross-correlate 1 Hz HR to find lag (seconds) that maximizes Pearson r.
    Positive result => shift TCX forward by +lag seconds (i.e., TCX starts earlier).
    """
    # CPET: already in seconds-from-start
    t_c, hr_c = _series_to_1hz(cpet["time_s"].to_numpy(), cpet["hr"].to_numpy(), smooth_sec)
    # TCX: resample HR vs seconds-from-start
    t_t, hr_t = _series_to_1hz(tcx["t_s"].to_numpy(), tcx["hr"].to_numpy(), smooth_sec)

    if len(hr_c) < 10 or len(hr_t) < 10:
        return None

    # Put both on a common grid relative to their own starts; we'll compare with integer lags
    # Build a dict for fast lookup
    hr_t_map = dict(zip(t_t.astype(int), hr_t))
    hr_c_map = dict(zip(t_c.astype(int), hr_c))

    # Candidate lags
    lags = np.arange(-max_shift, max_shift + 1, 1, dtype=int)
    best_lag, best_r = None, -np.inf

    # Build union of keys to speed loops
    ks_c = set(hr_c_map.keys())
    ks_t = set(hr_t_map.keys())

    for lag in lags:
        # Compare hr_c[t] with hr_t[t - lag]
        common = [k for k in ks_c if (k - lag) in ks_t]
        if len(common) < 20:
            continue
        a = np.array([hr_c_map[k] for k in common], dtype=float)
        b = np.array([hr_t_map[k - lag] for k in common], dtype=float)
        if np.std(a) < 1e-6 or np.std(b) < 1e-6:
            continue
        r = np.corrcoef(a, b)[0, 1]
        if r > best_r:
            best_r, best_lag = r, int(lag)

    return best_lag


def interpolate_power_onto_cpet(cpet: pd.DataFrame,
                                tcx: pd.DataFrame,
                                offset_s: int) -> pd.DataFrame:
    """
    Shift TCX by offset_s, then interpolate watts and tcx HR onto CPET time_s.
    """
    tcx_shift = tcx.copy()
    tcx_shift["t_shift_s"] = tcx_shift["t_s"] + offset_s

    # Interpolate watts and hr at CPET times
    t_src = tcx_shift["t_shift_s"].to_numpy()
    w_src = tcx_shift["watts"].to_numpy()
    h_src = tcx_shift["hr"].to_numpy()

    # Keep finite
    m_w = np.isfinite(t_src) & np.isfinite(w_src)
    m_h = np.isfinite(t_src) & np.isfinite(h_src)

    t_cpet = cpet["time_s"].to_numpy()

    watts_i = np.full_like(t_cpet, np.nan, dtype=float)
    hr_i = np.full_like(t_cpet, np.nan, dtype=float)

    if m_w.sum() >= 2:
        watts_i = np.interp(t_cpet, t_src[m_w], w_src[m_w], left=np.nan, right=np.nan)
    if m_h.sum() >= 2:
        hr_i = np.interp(t_cpet, t_src[m_h], h_src[m_h], left=np.nan, right=np.nan)

    out = cpet.copy()
    out["Power (W)"] = watts_i
    out["tcx_hr_interp"] = hr_i
    out["hr_diff_bpm"] = out["hr"] - out["tcx_hr_interp"]
    out["tcx_time_offset_s"] = float(offset_s)
    return out


# ---------- Fallback HR-nearest (optional) ----------

def hr_nearest_attach_power(cpet: pd.DataFrame,
                            tcx: pd.DataFrame,
                            bpm_window: float = 3.0,
                            time_window_s: float = 15.0,
                            offset_s: int = 0) -> pd.DataFrame:
    """
    For each CPET sample, find TCX sample within +/- bpm_window and +/- time_window_s (after offset).
    Attach its power; otherwise leave NaN.
    """
    tcx2 = tcx.copy()
    tcx2["t_shift_s"] = tcx2["t_s"] + offset_s

    out = cpet.copy()
    pw = np.full(len(out), np.nan, dtype=float)
    hr_i = np.full(len(out), np.nan, dtype=float)

    j = 0
    for i, (t, hr) in enumerate(zip(out["time_s"].to_numpy(), out["hr"].to_numpy())):
        # advance pointer near this time
        while j + 1 < len(tcx2) and tcx2.loc[j + 1, "t_shift_s"] < t - time_window_s:
            j += 1
        # search local window
        lo = j
        while lo > 0 and tcx2.loc[lo - 1, "t_shift_s"] >= t - time_window_s:
            lo -= 1
        hi = j
        while hi + 1 < len(tcx2) and tcx2.loc[hi + 1, "t_shift_s"] <= t + time_window_s:
            hi += 1
        window = tcx2.loc[lo:hi]
        if window.empty:
            continue
        # filter by HR window
        cand = window[np.abs(window["hr"] - hr) <= bpm_window]
        if cand.empty:
            continue
        # pick nearest in time
        idx = (cand["t_shift_s"] - t).abs().idxmin()
        pw[i] = float(cand.loc[idx, "watts"])
        hr_i[i] = float(cand.loc[idx, "hr"])

    out["Power (W)"] = pw
    out["tcx_hr_interp"] = hr_i
    out["hr_diff_bpm"] = out["hr"] - out["tcx_hr_interp"]
    out["tcx_time_offset_s"] = float(offset_s)
    return out


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Merge TCX power into CPET CSV using HR alignment.")
    ap.add_argument("--cpet", required=True, help="CPET CSV (needs 'Timer (s)' and 'HR (bpm)' columns)")
    ap.add_argument("--tcx", required=True, help="TCX file with HR and Watts")
    ap.add_argument("--out", required=True, help="Output merged CSV")
    ap.add_argument("--max-shift", type=int, default=120, help="Max absolute shift to search (s)")
    ap.add_argument("--smooth-sec", type=float, default=5.0, help="HR smoothing for alignment (s)")
    ap.add_argument("--offset", type=int, default=None, help="Override: fixed TCX->CPET shift (s)")
    ap.add_argument("--hr-nearest", action="store_true", help="Fallback: nearest TCX sample by HR/time window")
    ap.add_argument("--bpm-window", type=float, default=3.0, help="HR window for --hr-nearest (bpm)")
    ap.add_argument("--time-window", type=float, default=15.0, help="Time window for --hr-nearest (s)")

    args = ap.parse_args()

    cpet = read_cpet(args.cpet)
    tcx = read_tcx(args.tcx)

    if args.offset is None:
        lag = estimate_offset_by_hr(cpet, tcx, max_shift=args.max_shift, smooth_sec=args.smooth_sec)
        if lag is None:
            print("Couldn't estimate offset from HR; consider --offset or --hr-nearest.", flush=True)
            lag = 0
    else:
        lag = int(args.offset)

    print(f"Using TCX time offset: {lag} s (positive means TCX shifted forward).")

    if args.hr_nearest:
        merged = hr_nearest_attach_power(
            cpet, tcx, bpm_window=args.bpm_window, time_window_s=args.time_window, offset_s=lag
        )
    else:
        merged = interpolate_power_onto_cpet(cpet, tcx, offset_s=lag)

    merged.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with {len(merged)} rows.")


if __name__ == "__main__":
    main()
