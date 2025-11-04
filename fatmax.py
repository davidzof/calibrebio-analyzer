#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fat Max plot computed from VO2/VCO2 (Frayn 1983)
 © David George, 2025

- X axis: Heart Rate (bpm)
- Y axis: Fat oxidation (g/min), derived from VO2/VCO2
- Shows raw points (optional), per-BPM median curve, smoothed curve
- Highlights FatMax HR where smoothed fat oxidation is maximal

Args:
  --input FILE
  --trim a,b          Optional time trim as start,end in seconds (empty side = open range)
  --smooth-bpm N      Rolling window (in bpm) for smoothing the per-BPM curve (default: 5)
  --no-points         Hide raw scatter points, show only curves/marker
  --print             Print FatMax HR and value to stdout
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cpet_common import COLS, parse_trim  # COLS["HR"] -> "HR (bpm)"

def _fat_grams_per_min(vo2_lpm: pd.Series, vco2_lpm: pd.Series) -> pd.Series:
    """
    Frayn equations (assumes VO2/VCO2 in L/min, STPD):
      Fat (g/min) = 1.695*VO2 - 1.701*VCO2
    Clamp negatives to 0 (when RER > ~0.99), then return.
    """
    fat = 1.695 * vo2_lpm - 1.701 * vco2_lpm
    fat = pd.to_numeric(fat, errors="coerce")
    fat[fat < 0] = 0.0
    return fat

def run_fatmax(input_path: str, trim: str = "", smooth_bpm: int = 5, show_points: bool = True, do_print: bool = False):
    df = pd.read_csv(input_path)

    # Require just what we need
    needed = [COLS["HR"], COLS["VO2"], COLS["VCO2"], "Timer (s)"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required column(s) for Fat Max: {missing}")

    # Optional time trim
    if trim:
        try:
            df = parse_trim(df, trim)
        except Exception:
            raise SystemExit(f"Could not parse --trim '{trim}'. Expected 'start,end' in seconds.")

    # Extract series
    hr   = pd.to_numeric(df[COLS["HR"]],  errors="coerce")
    vo2  = pd.to_numeric(df[COLS["VO2"]], errors="coerce")   # slpm ~ L/min
    vco2 = pd.to_numeric(df[COLS["VCO2"]], errors="coerce")  # slpm ~ L/min

    mask = np.isfinite(hr) & np.isfinite(vo2) & np.isfinite(vco2)
    hr, vo2, vco2 = hr[mask], vo2[mask], vco2[mask]

    if hr.empty:
        raise SystemExit("No valid HR/VO2/VCO2 data to plot after filtering.")

    # Compute fat oxidation (g/min)
    fat_gpm = _fat_grams_per_min(vo2, vco2)

    # Build per-BPM (integer) aggregated curve, then smooth over BPM
    bpm = hr.round().astype("Int64")
    curve = (
        pd.DataFrame({"bpm": bpm, "fat": fat_gpm})
        .groupby("bpm", dropna=True)["fat"]
        .median()
        .sort_index()
        .astype(float)
    )
    if curve.empty:
        raise SystemExit("Not enough data to construct the per-BPM curve.")

    # Regularize BPM index so rolling window counts are consistent
    full_idx = pd.RangeIndex(int(curve.index.min()), int(curve.index.max()) + 1)
    curve = curve.reindex(full_idx).interpolate(limit_direction="both")

    if smooth_bpm and smooth_bpm > 1:
        smooth = curve.rolling(window=int(smooth_bpm), center=True, min_periods=max(1, smooth_bpm // 2)).mean()
    else:
        smooth = curve.copy()

    # FatMax = BPM at which smoothed fat oxidation peaks
    fatmax_bpm = int(smooth.idxmax())
    fatmax_val = float(smooth.loc[fatmax_bpm])

    # Plot
    plt.figure(figsize=(10, 6))
    if show_points:
        plt.scatter(hr, fat_gpm, s=12, alpha=0.35, label="Raw points (g/min)")

    plt.plot(curve.index, curve.values, linewidth=1.2, alpha=0.6, label="Per-BPM median (g/min)")
    plt.plot(smooth.index, smooth.values, linewidth=2.2, label=f"Smoothed ({smooth_bpm} bpm)")

    # Marker + guide for FatMax
    plt.axvline(fatmax_bpm, linestyle="--", alpha=0.6, label=f"FatMax HR ≈ {fatmax_bpm} bpm")
    plt.scatter([fatmax_bpm], [fatmax_val], s=50, zorder=5, label=f"Peak {fatmax_val:.2f} g/min")

    plt.xlabel("Heart Rate (bpm)")
    plt.ylabel("Fat oxidation (g/min)")
    plt.title("Fat Max from VO₂/VCO₂ (Frayn)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()

    if do_print:
        print(f"FatMax_HR_bpm,{fatmax_bpm}")
        print(f"FatMax_Fat_g_per_min,{fatmax_val:.4f}")

def main():
    ap = argparse.ArgumentParser(description="Fat Max from VO₂/VCO₂ (Frayn): Fat oxidation (g/min) vs HR")
    ap.add_argument("--input", required=True, help="CSV file")
    ap.add_argument("--trim", type=str, default="", help="Time trim as start,end in seconds (e.g., 60,900)")
    ap.add_argument("--smooth-bpm", type=int, default=5, help="Rolling window (bpm) for the smoothed curve")
    ap.add_argument("--no-points", action="store_true", help="Hide raw scatter points")
    ap.add_argument("--print", action="store_true", help="Print FatMax HR and value to stdout")
    args = ap.parse_args()

    run_fatmax(
        input_path=args.input,
        trim=args.trim,
        smooth_bpm=args.smooth_bpm,
        show_points=not args.no_points,
        do_print=args.print,
    )

if __name__ == "__main__":
    main()
