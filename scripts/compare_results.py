#!/usr/bin/env python3
"""
Compare two FastSim result pickles for exact equality.

FastSim is deterministic, so two runs of the same workload must produce
exactly equal job histories; any difference means a change altered
scheduling behaviour. Exits 0 when equal, 1 when not.

With --ignore-missing-columns only the columns present in both files are
compared (and the asymmetric columns are listed). This is for comparing
against result files written by older FastSim revisions that dumped a
different set of columns; leave it off otherwise.
"""

import argparse
import sys

import pandas as pd


def compare(path_a, path_b, ignore_missing_columns=False):
    df_a = pd.read_pickle(path_a)
    df_b = pd.read_pickle(path_b)

    cols_a, cols_b = list(df_a.columns), list(df_b.columns)
    if cols_a != cols_b:
        only_a = [c for c in cols_a if c not in cols_b]
        only_b = [c for c in cols_b if c not in cols_a]
        if not ignore_missing_columns:
            print(f"DIFFERENT: column mismatch (only in {path_a}: {only_a}, "
                  f"only in {path_b}: {only_b})")
            return 1
        print(f"Ignoring asymmetric columns (only in {path_a}: {only_a}, "
              f"only in {path_b}: {only_b})")
        common = [c for c in cols_a if c in cols_b]
        df_a, df_b = df_a[common], df_b[common]

    if df_a.shape != df_b.shape:
        print(f"DIFFERENT: shape {df_a.shape} vs {df_b.shape}")
        return 1

    if df_a.equals(df_b):
        print(f"EQUAL: {df_a.shape[0]} rows x {df_a.shape[1]} columns")
        return 0

    bad = [col for col in df_a.columns if not df_a[col].equals(df_b[col])]
    print(f"DIFFERENT: {len(bad)} column(s) differ: {bad}")
    for col in bad[:5]:
        neq = df_a[col].astype(str) != df_b[col].astype(str)
        rows = list(df_a.index[neq][:5])
        print(f"  {col}: {int(neq.sum())} differing row(s), first at {rows}")
    return 1


def main():
    parser = argparse.ArgumentParser(
        description="Assert two FastSim result pickles are exactly equal.")
    parser.add_argument("pickle_a")
    parser.add_argument("pickle_b")
    parser.add_argument("--ignore-missing-columns", action="store_true",
                        help="Compare only the columns present in both files")
    args = parser.parse_args()
    sys.exit(compare(args.pickle_a, args.pickle_b, args.ignore_missing_columns))


if __name__ == "__main__":
    main()
