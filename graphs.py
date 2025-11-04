#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Time-series CPET plots (multi-axis) with shared VO2/VCO2 axis and optional VT markers.
 © David George, 2025

Args:
  --input FILE
  --smooth-sec N       Centered rolling smoothing window in seconds (0 = none)
  --show ...           Metrics to plot (aliases OK, order respected): HR VE VO2 VCO2 VeqO2 VeqCO2 RER Power
  --trim a,b           Time trim as start,end in seconds (empty side = open range)
  --mark-vt            Mark VT1/VT2 using V-slope (LV1) + VE/VCO2 nadir (LV2) and annotate Peak VO2

VO2/VCO2 per kg mode is unchanged (see bottom).
"""
from itertools import cycle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from cpet_common import (
    COLS, to_canonical, seconds_to_rows, smooth_series,
    parse_trim, validate_required
)

# --- add local safe aliases so Power works even if COLS doesn't know it ---
COLS.setdefault("Power", "Power")              # new header you mentioned
# Optional legacy alias, harmless if absent:
COLS.setdefault("CyclingPower", "Cycling Power (W)")

# ---------- helpers specific to per-kg plotting ----------
def _get_first_present(df: pd.DataFrame, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"None of the expected columns found: {candidates}")
    return None

def _pick_x_col(df: pd.DataFrame, x_mode: str):
    if x_mode == "time":
        return "Timer (s)"
    elif x_mode == "hr":
        return COLS["HR"]
    raise ValueError("--x must be 'time' or 'hr'")

def _nadir_then_rise(arr: np.ndarray, win: int = 7):
    """(fallback) Return index of smoothed minimum if followed by a rise; else None."""
    s = pd.Series(arr)
    v = s.rolling(win, center=True, min_periods=max(1, win//2)).median()
    if v.isna().all():
        return None
    try:
        idx_min = int(np.nanargmin(v.to_numpy()))
    except ValueError:
        return None
    after = v.iloc[idx_min + win : idx_min + 3*win]
    if len(after) == 0 or np.isnan(v.iloc[idx_min]):
        return None
    ok = (np.nanmean(after) > v.iloc[idx_min])
    return idx_min if ok else None

# ---------- Advanced VT detection (used by --mark-vt) ----------
def _detect_vt_advanced(df: pd.DataFrame, smooth_sec: float):
    """
    Returns dict with keys 'lv1', 'lv2', 'peak_vo2' mapping to dicts
    {idx, time, hr, vo2, vco2}. Uses:
      - LV1: V-slope (piecewise linear breakpoint VCO2 vs VO2)
      - LV2: VE/VCO2 nadir after LV1
      - Peak VO2: max VO2, but restricted to <900 s if data extends past 900 s
    Falls back to None if detection fails.
    """
    try:
        t = pd.to_numeric(df["Timer (s)"], errors="coerce")
        vo2 = pd.to_numeric(df[COLS["VO2"]], errors="coerce")
        vco2 = pd.to_numeric(df[COLS["VCO2"]], errors="coerce")
        ve = pd.to_numeric(df[COLS["VE"]], errors="coerce")
        hr = pd.to_numeric(df[COLS["HR"]], errors="coerce")

        # smoothing
        win = max(5, seconds_to_rows(t, smooth_sec))
        vo2s = smooth_series(vo2, win)
        vco2s = smooth_series(vco2, win)
        ves   = smooth_series(ve, win)

        valid = pd.DataFrame({"x": vo2s, "y": vco2s}).dropna()
        out = {}

        # ---- LV1 via V-slope breakpoint ----
        if len(valid) >= 30:
            x = valid["x"].to_numpy()
            y = valid["y"].to_numpy()
            n = len(valid)
            best_i, best_err = None, np.inf
            for i in range(10, n-10):
                a1, b1 = np.polyfit(x[:i], y[:i], 1)
                a2, b2 = np.polyfit(x[i:], y[i:], 1)
                err = ((y[:i] - (a1*x[:i]+b1))**2).sum() + ((y[i:] - (a2*x[i:]+b2))**2).sum()
                if err < best_err:
                    best_err, best_i = err, i
            if best_i is not None:
                vo2_bp = float(x[best_i])
                idx1 = (vo2s - vo2_bp).abs().idxmin()
                out["lv1"] = {
                    "idx": int(idx1),
                    "time": float(t.iloc[idx1]),
                    "hr": int(round(hr.iloc[idx1])) if np.isfinite(hr.iloc[idx1]) else None,
                    "vo2": float(vo2s.iloc[idx1]),
                    "vco2": float(vco2s.iloc[idx1]),
                }

        # ---- LV2 via VE/VCO2 nadir after LV1 ----
        if "lv1" in out:
            ve_vco2 = (ves / vco2s).replace([np.inf, -np.inf], np.nan)
            # avoid final recovery: ignore last 120 s if possible
            end_cut_idx = len(df) - max(1, seconds_to_rows(t, 120))
            search = ve_vco2.iloc[out["lv1"]["idx"]: end_cut_idx].dropna()
            if not search.empty:
                idx2 = int(search.idxmin())
                out["lv2"] = {
                    "idx": idx2,
                    "time": float(t.iloc[idx2]),
                    "hr": int(round(hr.iloc[idx2])) if np.isfinite(hr.iloc[idx2]) else None,
                    "vo2": float(vo2s.iloc[idx2]),
                    "vco2": float(vco2s.iloc[idx2]),
                }

        # ---- Peak VO2 (pre-glitch if data past 900 s) ----
        if (t > 900).any():
            mask = t < 900
        else:
            mask = np.isfinite(t)
        if np.any(mask):
            idxp = int(vo2[mask].idxmax())
            out["peak_vo2"] = {
                "idx": idxp,
                "time": float(t.iloc[idxp]),
                "hr": int(round(hr.iloc[idxp])) if np.isfinite(hr.iloc[idxp]) else None,
                "vo2": float(vo2.iloc[idxp]),
                "vco2": float(vco2.iloc[idxp]) if np.isfinite(vco2.iloc[idxp]) else None,
            }
        return out
    except Exception:
        return {}

# ---------- original multi-axis plotting ----------
def run_graphs(input_path: str, smooth_sec: float = 0.0, show=None, trim: str = "", mark_vt: bool = False):
    df = pd.read_csv(input_path)
    validate_required(df)

    # Optional time trimming
    if trim:
        try:
            df = parse_trim(df, trim)
        except Exception:
            raise SystemExit(f"Could not parse --trim '{trim}'. Expected 'start,end' in seconds.")

    # Determine which metrics to show and in what order
    default_order = ["HR", "VE", "VO2", "VCO2", "VeqO2", "VeqCO2", "RER"]
    order = to_canonical(show) if show else default_order[:]

    # Keep only metrics whose mapped column exists AND also allow raw column name 'Power'
    # (in case to_canonical passes through "Power" as-is).
    # First, ensure COLS has a mapping for 'Power' -> 'Power' (done above).
    order = [k for k in order if (k in COLS and COLS[k] in df.columns) or (k not in COLS and k in df.columns)]
    if not order:
        raise SystemExit("None of the requested metrics were found in the file.")

    # Smoothing window in rows
    win = seconds_to_rows(df["Timer (s)"], smooth_sec)

    # Prepare series for selected metrics (support raw 'Power' key too)
    def _series_for(name: str):
        col = COLS.get(name, name)  # fallback to raw name
        return smooth_series(df[col], win)

    series = {name: _series_for(name) for name in order}

    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    axes = [ax1]
    metric_axes = {}
    t = pd.to_numeric(df["Timer (s)"], errors="coerce")
    hr_vals = pd.to_numeric(df[COLS["HR"]], errors="coerce").to_numpy()

    base_colors = plt.rcParams.get("axes.prop_cycle").by_key().get("color", [])
    if not base_colors:
        base_colors = [f"C{i}" for i in range(10)]
    color_cycle = cycle(base_colors)
    color_map = {k: next(color_cycle) for k in order}

    # Primary axis
    primary = "HR" if "HR" in order else order[0]
    ax1.plot(t, series[primary], linewidth=2, label=primary, color=color_map[primary])
    ax1.set_xlabel("Time (s)")
    # y-label with special cases
    if primary == "VE":
        ylab_primary = "VE (L/min)"
    elif primary == "Power":
        ylab_primary = "Power (W)"
    else:
        ylab_primary = COLS.get(primary, primary)
    ax1.set_ylabel(ylab_primary, color=color_map[primary])
    ax1.tick_params(axis="y", labelcolor=color_map[primary])
    metric_axes[primary] = ax1

    # If both VO2 & VCO2 are requested, plot them on the SAME y-axis
    pair_present = ("VO2" in order) and ("VCO2" in order)
    pair_axis = None

    remaining = [k for k in order if k != primary]
    step, offset = 60, 0

    for k in remaining:
        # Shared VO2/VCO2 axis
        if pair_present and k in ("VO2", "VCO2"):
            if pair_axis is None:
                ax = ax1.twinx()
                if offset:
                    ax.spines.right.set_position(("outward", offset))
                axes.append(ax)
                pair_axis = ax
                pair_axis.set_ylabel("VO2 & VCO2 (L/min)")
                offset += step
            pair_axis.plot(t, series[k], linewidth=2, label=k, alpha=0.9, color=color_map[k])
            metric_axes[k] = pair_axis
            continue

        # Default behavior for other metrics
        ax = ax1.twinx()
        if offset:
            ax.spines.right.set_position(("outward", offset))
        axes.append(ax)
        ax.plot(t, series[k], linewidth=2, label=k, alpha=0.9, color=color_map[k])
        # ylabel with special cases
        if k == "VE":
            ylabel = "VE (L/min)"
        elif k == "Power":
            ylabel = "Power (W)"
        else:
            ylabel = COLS.get(k, k)
        ax.set_ylabel(ylabel, color=color_map[k])
        ax.tick_params(axis="y", labelcolor=color_map[k])
        metric_axes[k] = ax
        offset += step

    # Lock identical limits for the shared VO2/VCO2 axis
    if pair_axis is not None:
        vmax = np.nanmax([np.nanmax(series["VO2"]), np.nanmax(series["VCO2"])])
        if not np.isfinite(vmax):
            vmax = 1.0
        ymin, ymax = 0.0, float(vmax) * 1.05
        pair_axis.set_ylim(ymin, ymax)

    # Combined legend
    lines, labels = [], []
    for ax in axes:
        l, lab = ax.get_legend_handles_labels()
        lines += l; labels += lab
    ax1.legend(lines, labels, loc="upper left")

    title = "CPET metrics over time"
    if seconds_to_rows(df["Timer (s)"], smooth_sec) and seconds_to_rows(df["Timer (s)"], smooth_sec) > 1:
        title += f"  (smoothed ~{int(round(smooth_sec))}s)"
    plt.title(title)

    # ---- VT markers (advanced method with graceful fallback) ----
    if mark_vt:
        vt = _detect_vt_advanced(df, smooth_sec)

        def _vline(x, txt, yfrac=0.95):
            ax1.axvline(x, linestyle="--", alpha=0.65)
            ax1.text(x, ax1.get_ylim()[1]*yfrac, txt, ha="left", va="top", rotation=90, fontsize=9)

        if "lv1" in vt:
            _vline(vt["lv1"]["time"], f"VT1 ≈ {vt['lv1']['hr']} bpm")
        if "lv2" in vt:
            _vline(vt["lv2"]["time"], f"VT2 ≈ {vt['lv2']['hr']} bpm", yfrac=0.90)
        if "peak_vo2" in vt:
            _vline(vt["peak_vo2"]["time"], f"Peak VO2 ≈ {vt['peak_vo2']['hr']} bpm", yfrac=0.85)

        # Fallback to ventilatory equivalents if advanced detection failed
        need_fallback = ("lv1" not in vt) and ("lv2" not in vt)
        if need_fallback and "VeqO2" in series and "VeqCO2" in series:
            veqo2 = pd.to_numeric(series["VeqO2"], errors="coerce").to_numpy()
            veqco2 = pd.to_numeric(series["VeqCO2"], errors="coerce").to_numpy()
            t_vals = t.to_numpy()
            vt1_idx = _nadir_then_rise(veqo2, win=7)
            vt2_idx = _nadir_then_rise(veqco2, win=7)
            if vt1_idx is not None and 0 <= vt1_idx < len(t_vals):
                _vline(t_vals[vt1_idx], f"VT1 ≈ {int(round(hr_vals[vt1_idx]))} bpm")
            if vt2_idx is not None and 0 <= vt2_idx < len(t_vals):
                _vline(t_vals[vt2_idx], f"VT2 ≈ {int(round(hr_vals[vt2_idx]))} bpm", yfrac=0.90)

    plt.tight_layout()
    plt.show()

# ---------- VO2/VCO2 per kg plotting ----------
def run_vo2kg(input_path: str,
              smooth_sec: float = 0.0,
              trim: str = "",
              x_mode: str = "time",
              show_vo2: bool = True,
              show_vco2: bool = False,
              title: str | None = None,
              mark_peaks: bool = False):
    """
    Plot VO2/kg (ml/kg/min) and/or VCO2/kg (ml/kg/min) vs time or HR.
    """
    df = pd.read_csv(input_path)
    validate_required(df)

    if trim:
        try:
            df = parse_trim(df, trim)
        except Exception:
            raise SystemExit(f"Could not parse --trim '{trim}'. Expected 'start,end' in seconds.")

    vo2kg_col = _get_first_present(df, ["VO2/kg (ml/kg/min)", "VO2_kg", "VO2_per_kg", "VO2/kg"])
    vco2kg_col = _get_first_present(df, ["VCO2/kg (ml/kg/min)", "VCO2_kg", "VCO2_per_kg", "VCO2/kg"], required=False)

    x_col = _pick_x_col(df, x_mode)
    win = seconds_to_rows(df["Timer (s)"], smooth_sec)

    vo2kg = smooth_series(pd.to_numeric(df[vo2kg_col], errors="coerce"), win) if show_vo2 else None
    vco2kg = smooth_series(pd.to_numeric(df[vco2kg_col], errors="coerce"), win) if (show_vco2 and vco2kg_col) else None

    x_vals = pd.to_numeric(df[x_col], errors="coerce").to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    lines, labels = [], []

    if show_vo2 and vo2kg is not None:
        l1, = ax.plot(x_vals, vo2kg, label="VO2/kg (ml·kg⁻¹·min⁻¹)", linewidth=2)
        lines.append(l1); labels.append("VO2/kg")
        if mark_peaks and not np.isnan(vo2kg.to_numpy()).all():
            idx = int(np.nanargmax(vo2kg.to_numpy()))
            if np.isfinite(vo2kg.iloc[idx]):
                ax.scatter([x_vals[idx]], [vo2kg.iloc[idx]])
                ax.annotate(f"VO2max/kg ≈ {vo2kg.iloc[idx]:.1f}",
                            (x_vals[idx], vo2kg.iloc[idx]), textcoords="offset points", xytext=(6,6))

    if show_vco2 and vco2kg is not None:
        l2, = ax.plot(x_vals, vco2kg, label="VCO2/kg (ml·kg⁻¹·min⁻¹)", linewidth=2, alpha=0.9)
        lines.append(l2); labels.append("VCO2/kg")
        if mark_peaks and not np.isnan(vco2kg.to_numpy()).all():
            idx2 = int(np.nanargmax(vco2kg.to_numpy()))
            if np.isfinite(vco2kg.iloc[idx2]):
                ax.scatter([x_vals[idx2]], [vco2kg.iloc[idx2]])
                ax.annotate(f"VCO2peak/kg ≈ {vco2kg.iloc[idx2]:.1f}",
                            (x_vals[idx2], vco2kg.iloc[idx2]), textcoords="offset points", xytext=(6,6))

    ax.set_xlabel("Time (s)" if x_mode == "time" else "Heart rate (bpm)")
    ax.set_ylabel("ml·kg⁻¹·min⁻¹")
    ttl = title or f"{' & '.join([lbl for lbl in labels])} vs {'Time' if x_mode=='time' else 'HR'}"
    if smooth_sec and smooth_sec > 0:
        ttl += f"  (smoothed ~{int(round(smooth_sec))}s)"
    ax.set_title(ttl)
    ax.grid(True, alpha=0.25)
    if lines:
        ax.legend()

    plt.tight_layout()
    plt.show()

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="CPET time-series plotter (exact headers)")
    ap.add_argument("--input", required=True, help="CSV file")
    ap.add_argument("--smooth-sec", type=float, default=0.0, help="Centered rolling smoothing window in seconds (0 = none)")
    ap.add_argument("--show", nargs="+", help="Metrics to plot (aliases OK, order respected): HR VE VO2 VCO2 VeqO2 VeqCO2 RER Power")
    ap.add_argument("--trim", type=str, default="", help="Time trim as start,end in seconds (e.g., 5,1000). Empty side means open range.")
    ap.add_argument("--mark-vt", action="store_true", help="Mark VT1/VT2 (V-slope + VE/VCO2) and Peak VO2")

    # per-kg plotting flags
    ap.add_argument("--vo2kg", action="store_true", help="Plot VO2/kg and/or VCO2/kg vs time or HR")
    ap.add_argument("--x", choices=["time", "hr"], default="time", help="X-axis for --vo2kg (default: time)")
    ap.add_argument("--show-vo2kg", action="store_true", help="(VO2/kg) Show VO2/kg")
    ap.add_argument("--show-vco2kg", action="store_true", help="(VO2/kg) Show VCO2/kg")
    ap.add_argument("--both", action="store_true", help="(VO2/kg) Show both VO2/kg and VCO2/kg")
    ap.add_argument("--title", type=str, default=None, help="(VO2/kg) Custom title")
    ap.add_argument("--mark-peaks", action="store_true", help="(VO2/kg) Annotate peak (max) values")

    args = ap.parse_args()

    if args.vo2kg:
        show_vo2 = args.show_vo2kg or args.both or (not args.show_vco2kg and not args.both)
        show_vco2 = args.show_vco2kg or args.both
        run_vo2kg(
            input_path=args.input,
            smooth_sec=args.smooth_sec,
            trim=args.trim,
            x_mode=args.x,
            show_vo2=show_vo2,
            show_vco2=show_vco2,
            title=args.title,
            mark_peaks=args.mark_peaks
        )
        return

    run_graphs(
        input_path=args.input,
        smooth_sec=args.smooth_sec,
        show=args.show,
        trim=args.trim,
        mark_vt=args.mark_vt,
    )

if __name__ == "__main__":
    main()
