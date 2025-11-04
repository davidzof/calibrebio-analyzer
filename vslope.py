#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPET V-slope (VCO₂ vs VO₂) plotter (EXACT headers) with optional VT1 detection.
 © David George, 2025

Args:
  --input FILE
  --smooth-sec N      Centered rolling smoothing window in seconds (0 = none)
  --trim a,b          Time trim as start,end in seconds (empty side = open range)
  --color {HR,time}   Color points by HR or time (default: HR)
  --no-thresholds     Disable VT1 detection/annotation

Always displays the figure; no saving.
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cpet_common import (
    COLS, seconds_to_rows, parse_trim, validate_required
)

# ---------- VT1 (V-slope breakpoint) helpers ----------
def _two_line_sse(x, y, k):
    """Fit two least-squares lines split at index k; return SSE and segments."""
    x1, y1 = x[:k+1], y[:k+1]
    x2, y2 = x[k+1:], y[k+1:]
    if len(x1) < 3 or len(x2) < 3:
        return np.inf, None, None
    A1 = np.vstack([x1, np.ones_like(x1)]).T
    A2 = np.vstack([x2, np.ones_like(x2)]).T
    m1, b1 = np.linalg.lstsq(A1, y1, rcond=None)[0]
    m2, b2 = np.linalg.lstsq(A2, y2, rcond=None)[0]
    sse = np.sum((y1 - (m1*x1+b1))**2) + np.sum((y2 - (m2*x2+b2))**2)
    return sse, (m1, b1), (m2, b2)

def _vslope_breakpoint(x, y, min_gap=20):
    """Brute-force breakpoint search with edge gap. Returns (k, seg1, seg2)."""
    n = len(x)
    best = (np.inf, None, None, None)
    for k in range(min_gap, n - min_gap):
        sse, seg1, seg2 = _two_line_sse(x, y, k)
        if sse < best[0]:
            best = (sse, k, seg1, seg2)
    _, k, seg1, seg2 = best
    return k, seg1, seg2
# -----------------------------------------------------


def run_vslope(input_path: str,
               smooth_sec: float = 0.0,
               trim: str = "",
               color: str = "HR",
               detect_thresholds: bool = True):
    df = pd.read_csv(input_path)
    validate_required(df)

    # Optional time trimming
    if trim:
        try:
            df = parse_trim(df, trim)
        except Exception:
            raise SystemExit(f"Could not parse --trim '{trim}'. Expected 'start,end' in seconds.")

    # Smoothing window in rows
    win = seconds_to_rows(df["Timer (s)"], smooth_sec)

    # Prepare smoothed VO2, VCO2
    vo2 = pd.to_numeric(df[COLS["VO2"]], errors="coerce")
    vco2 = pd.to_numeric(df[COLS["VCO2"]], errors="coerce")
    if win and win > 1:
        vo2 = vo2.rolling(win, center=True, min_periods=max(1, win//3)).median()
        vco2 = vco2.rolling(win, center=True, min_periods=max(1, win//3)).median()

    # Color field (HR or time)
    if color == "HR":
        cvals = pd.to_numeric(df[COLS["HR"]], errors="coerce")
        clabel = "HR (bpm)"
    else:
        cvals = pd.to_numeric(df["Timer (s)"], errors="coerce")
        clabel = "Time (s)"
    if win and win > 1:
        cvals = cvals.rolling(win, center=True, min_periods=max(1, win//3)).median()

    # Mask and arrays
    mask = np.isfinite(vo2) & np.isfinite(vco2) & np.isfinite(cvals) & (vo2 > 0) & (vco2 > 0)
    x = vo2[mask].to_numpy()
    y = vco2[mask].to_numpy()
    c = cvals[mask].to_numpy()
    hr_all = pd.to_numeric(df[COLS["HR"]], errors="coerce")
    hr = hr_all[mask].to_numpy()

    plt.figure(figsize=(8, 8))
    sc = plt.scatter(x, y, c=c, s=16, alpha=0.8, cmap="viridis")
    cb = plt.colorbar(sc)
    cb.set_label(clabel)

    # 1:1 guideline
    if x.size and y.size:
        lim = max(np.nanmax(x), np.nanmax(y)) * 1.02
        plt.plot([0, lim], [0, lim], "--", color="gray", linewidth=1, alpha=0.6, label="VCO₂ = VO₂")

    # Optional VT1 detection via V-slope breakpoint
    vt1_hr = None
    if detect_thresholds and len(x) >= 40:
        order = np.argsort(x)  # ensure increasing VO2
        x_sorted, y_sorted = x[order], y[order]
        hr_sorted = hr[order]
        min_gap = max(10, len(x_sorted)//20)
        k, seg1, seg2 = _vslope_breakpoint(x_sorted, y_sorted, min_gap=min_gap)
        if k is not None:
            vt1_hr = float(hr_sorted[k])
            # Mark breakpoint in data-space (VO2,VCO2)
            bp_x, bp_y = x_sorted[k], y_sorted[k]
            plt.scatter([bp_x], [bp_y], s=60, zorder=6,
                        label=f"VT1 (V-slope) ≈ {int(round(vt1_hr))} bpm")

    plt.xlabel("VO₂ (L/min)")
    plt.ylabel("VCO₂ (L/min)")
    title = "V-slope (VCO₂ vs VO₂) colored by " + ("HR" if color == "HR" else "time")
    if win and win > 1:
        title += f"  | smoothed ~{int(round(smooth_sec))}s"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="upper left")
    plt.tight_layout()
    plt.show()

    # Optionally print VT1 to stdout in a machine-friendly way (keep quiet by default)
    # if vt1_hr is not None:
    #     print(f"VT1_HR_bpm,{vt1_hr:.0f}")


def main():
    ap = argparse.ArgumentParser(description="CPET V-slope plotter (exact headers)")
    ap.add_argument("--input", required=True, help="CSV file")
    ap.add_argument("--smooth-sec", type=float, default=0.0, help="Centered rolling smoothing window in seconds (0 = none)")
    ap.add_argument("--trim", type=str, default="", help="Time trim as start,end in seconds (e.g., 5,1000). Empty side means open range.")
    ap.add_argument("--color", choices=["HR","time"], default="HR", help="Color points by HR or time")
    ap.add_argument("--no-thresholds", action="store_true", help="Disable VT1 detection/annotation")
    args = ap.parse_args()

    run_vslope(
        input_path=args.input,
        smooth_sec=args.smooth_sec,
        trim=args.trim,
        color=args.color,
        detect_thresholds=not args.no_thresholds,
    )

if __name__ == "__main__":
    main()

