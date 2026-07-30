#!/usr/bin/env python
"""
Run the protein fitness prediction benchmark.

Usage:
    # Quick demo on synthetic data (no download needed, runs anywhere):
    python run_benchmark.py --demo

    # Real assays: point at a directory of ProteinGym-format CSVs
    # (see README.md for how to get these):
    python run_benchmark.py --data-dir data/DMS_ProteinGym_substitutions --limit 10
"""
from __future__ import annotations

import argparse
import glob
import os

from src.data import load_assay_csv, make_synthetic_assay
from src.evaluate import evaluate_many


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default=None, help="Directory of ProteinGym-format assay CSVs")
    parser.add_argument("--demo", action="store_true", help="Run on synthetic demo assays instead of real data")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N assays (real data is slow)")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--out", type=str, default="results.csv")
    args = parser.parse_args()

    if args.demo or not args.data_dir:
        print("Running on synthetic demo assays (use --data-dir for real ProteinGym data)...\n")
        assays = [make_synthetic_assay(seed=s, name=f"synthetic_{s}") for s in range(5)]
    else:
        paths = sorted(glob.glob(os.path.join(args.data_dir, "*.csv")))
        if args.limit:
            paths = paths[: args.limit]
        if not paths:
            raise SystemExit(f"No CSVs found in {args.data_dir}")
        print(f"Loading {len(paths)} assays from {args.data_dir}...\n")
        assays = [load_assay_csv(p) for p in paths]

    results = evaluate_many(assays, n_folds=args.n_folds)
    print(results.to_string(index=False))
    results.to_csv(args.out, index=False)
    print(f"\nSaved results to {args.out}")


if __name__ == "__main__":
    main()
