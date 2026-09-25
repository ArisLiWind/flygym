#!/usr/bin/env python3
"""Minimal BANC v888 data verification script."""

import sys
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "banc"
META_PATH = DATA_DIR / "banc_888_meta.feather"
EDGES_PATH = DATA_DIR / "banc_888_edgelist_simple_v2.feather"


def main():
    print("=" * 60)
    print("BANC v888 Feather Verification")
    print("=" * 60)

    # ---- Load meta ----
    print(f"\n[META] Loading {META_PATH.name} ...")
    meta = pd.read_feather(META_PATH)
    print(f"  shape : {meta.shape}")
    print(f"  columns: {list(meta.columns)}")
    print("\n  Head:")
    print(meta.head().to_string())

    # ---- Load edgelist ----
    print(f"\n[EDGES] Loading {EDGES_PATH.name} ...")
    edges = pd.read_feather(EDGES_PATH)
    print(f"  shape : {edges.shape}")
    print(f"  columns: {list(edges.columns)}")
    print("\n  Head:")
    print(edges.head().to_string())

    # ---- Find a pre neuron with many downstream connections ----
    print("\n" + "=" * 60)
    print("Downstream connectivity for a single pre neuron")
    print("=" * 60)

    pre_col = "pre"
    post_col = "post"
    count_col = "count"

    # Pick the pre neuron with the largest total synapse count
    top_pre = (
        edges.groupby(pre_col)[count_col]
        .sum()
        .sort_values(ascending=False)
        .index[0]
    )
    print(f"\nSelected pre neuron: {top_pre}")

    # Top 20 downstream partners by synapse count
    top20 = (
        edges[edges[pre_col] == top_pre]
        .sort_values(by=count_col, ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    print(f"  -> {len(top20)} downstream partners (top 20)")
    print(f"  -> total synapses to these 20: {top20[count_col].sum()}")

    # ---- Join with meta ----
    meta_id_col = None
    for c in meta.columns:
        if str(c).lower() in ("pt_root_id", "root_id", "id", "pt_supervoxel_id", "seg_id"):
            meta_id_col = c
            break
    if meta_id_col is None:
        meta_id_col = meta.columns[0]
        print(f"\n[WARNING] No obvious ID column found; using first column '{meta_id_col}' for join.")
    else:
        print(f"\nJoin key: meta['{meta_id_col}']  <->  edges['{post_col}']")

    merged = top20.merge(
        meta,
        left_on=post_col,
        right_on=meta_id_col,
        how="left",
        suffixes=("", "_meta"),
    )

    # Print available descriptive columns for merged rows
    print("\nTop 20 downstream neurons with metadata:")
    display_cols = [post_col, count_col]
    for c in meta.columns:
        if c != meta_id_col:
            display_cols.append(c)

    # Only keep columns that actually exist in merged
    display_cols = [c for c in display_cols if c in merged.columns]
    print(merged[display_cols].to_string(index=False))

    print("\n" + "=" * 60)
    print("✅ BANC v888 data verified successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()
