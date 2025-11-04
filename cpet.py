#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
 Front-end for plotting CPET data downloaded from CalibreBio portal
 © David George, 2025

Usage examples:
  # Default graphs (multi-axis, shared VO2/VCO2 scale)
  cpet.py --input FILE --smooth-sec 3 --show HR VE VO2 VCO2 --trim 5,900 --mark-vt

  # VSlope
  cpet.py --input FILE --vslope --smooth-sec 5 --trim 10,1000 --color HR

  # FatMax
  cpet.py --input FILE --fatmax --trim 60, --smooth-bpm 7 --no-points

  # VO2/VCO2 per kg
  cpet.py --input FILE --vo2kg --x hr --both --mark-peaks --title "Mass-specific O2/CO2 vs HR"

Notes:
  --mark-vt (graphs) now detects VT1 via the V-slope breakpoint (VCO2 vs VO2),
  VT2 via the VE/VCO2 nadir after VT1, and annotates Peak VO2 (pre-900 s if present).
"""
import argparse
import sys

from graphs import run_graphs, run_vo2kg
from vslope import run_vslope
from fatmax import run_fatmax


def build_parser():
    ap = argparse.ArgumentParser(description="CPET plots front end (flat flags: --vslope | --fatmax | --vo2kg)")

    # Common / graphs defaults
    ap.add_argument("--input", required=True, help="CSV file path")

    # ---- Global-ish options used by multiple modes ----
    ap.add_argument("--trim", type=str, default="",
                    help="Time trim as start,end in seconds (e.g., 5,1000). Empty side = open range.")
    ap.add_argument("--smooth-sec", type=float, default=0.0,
                    help="(graphs/vslope) Centered rolling smoothing window in seconds (0 = none)")

    # ---- Mode selectors (pick none for default graphs, or exactly one of the three below) ----
    ap.add_argument("--vslope", action="store_true", help="Run V-slope scatter (VCO2 vs VO2)")
    ap.add_argument("--fatmax", action="store_true", help="Run FatMax (fat oxidation vs HR)")
    ap.add_argument("--vo2kg", action="store_true", help="Run VO2/kg and/or VCO2/kg vs time or HR")

    # ---- Graphs (default mode) options ----
    ap.add_argument("--show", nargs="+",
                    help="(graphs) Metrics to plot (aliases OK): HR VE VO2 VCO2 VeqO2 VeqCO2 RER")
    ap.add_argument("--mark-vt", action="store_true",
                    help="(graphs) Mark VT1/VT2 (V-slope + VE/VCO2) and Peak VO2")

    # ---- VSlope options ----
    ap.add_argument("--color", choices=["HR", "time"], default="HR",
                    help="(vslope) Color by HR or time")
    ap.add_argument("--no-thresholds", action="store_true",
                    help="(vslope) Disable VT1 detection/annotation")

    # ---- FatMax options ----
    ap.add_argument("--smooth-bpm", type=int, default=5,
                    help="(fatmax) Smoothing window in bpm")
    ap.add_argument("--no-points", action="store_true",
                    help="(fatmax) Hide raw points")

    # ---- VO2/kg options ----
    ap.add_argument("--x", choices=["time", "hr"], default="time",
                    help="(vo2kg) X-axis: time or hr (default: time)")
    ap.add_argument("--show-vo2kg", action="store_true", help="(vo2kg) Show VO2/kg")
    ap.add_argument("--show-vco2kg", action="store_true", help="(vo2kg) Show VCO2/kg")
    ap.add_argument("--both", action="store_true", help="(vo2kg) Show both VO2/kg and VCO2/kg")
    ap.add_argument("--title", type=str, default=None, help="(vo2kg) Custom plot title")
    ap.add_argument("--mark-peaks", action="store_true", help="(vo2kg) Annotate peak (max) values")

    return ap


def _validate_mode(args: argparse.Namespace):
    selected = sum([args.vslope, args.fatmax, args.vo2kg])
    if selected > 1:
        print("Error: pick at most one of --vslope, --fatmax, or --vo2kg (or none for default graphs).", file=sys.stderr)
        sys.exit(2)


def main():
    ap = build_parser()
    args = ap.parse_args()
    _validate_mode(args)

    # Dispatch
    if args.vslope:
        # run_vslope(input_path, smooth_sec, trim, color, detect_thresholds=True)
        run_vslope(args.input, args.smooth_sec, args.trim, args.color, detect_thresholds=not args.no_thresholds)
        return

    if args.fatmax:
        # run_fatmax(input_path, trim, smooth_bpm, show_points[, print_stats])
        run_fatmax(args.input, args.trim, args.smooth_bpm, not args.no_points)
        return

    if args.vo2kg:
        # Default behavior: if no series flags, show VO2/kg only
        show_vo2 = args.show_vo2kg or args.both or (not args.show_vco2kg and not args.both)
        show_vco2 = args.show_vco2kg or args.both
        # run_vo2kg(input_path, smooth_sec, trim, x_mode, show_vo2, show_vco2, title, mark_peaks)
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

    # Default graphs (no mode flag)
    # run_graphs(input_path, smooth_sec, show, trim, mark_vt)
    run_graphs(args.input, args.smooth_sec, args.show, args.trim, mark_vt=args.mark_vt)


if __name__ == "__main__":
    main()
