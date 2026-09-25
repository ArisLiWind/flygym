#!/usr/bin/env python3
"""Honest BANC connectivity verification + immediate neural dynamics."""

import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "banc"
meta = pd.read_feather(DATA / "banc_888_meta.feather")
edges = pd.read_feather(DATA / "banc_888_edgelist_simple_v2.feather")

print("=" * 70)
print("1. HONEST JOIN CHECK")
print("=" * 70)

# Pick the pre neuron with highest total output synapse count
pre_totals = edges.groupby("pre")["count"].sum().sort_values(ascending=False)
top_pre = pre_totals.index[0]
print(f"Top pre neuron by total output count: {top_pre}")
print(f"  Total output synapses: {pre_totals.iloc[0]}")
print(f"  Unique downstream partners: {edges[edges['pre']==top_pre]['post'].nunique()}")

# Is this pre in meta?
in_meta = top_pre in set(meta["root_id"])
print(f"  Pre neuron in meta: {in_meta}")
if in_meta:
    row = meta[meta["root_id"] == top_pre].iloc[0]
    print(f"    cell_type   = {row.get('cell_type', 'N/A')}")
    print(f"    region      = {row.get('region', 'N/A')}")
    print(f"    super_class = {row.get('super_class', 'N/A')}")
    print(f"    nt_pred     = {row.get('neurotransmitter_predicted', 'N/A')}")

# Top 20 downstream, check join success
top20 = (
    edges[edges["pre"] == top_pre]
    .sort_values("count", ascending=False)
    .head(20)
    .reset_index(drop=True)
)

meta_ids = set(meta["root_id"])
found = 0
for _, r in top20.iterrows():
    post = r["post"]
    if post in meta_ids:
        found += 1

print(f"\nTop 20 downstream neurons:")
print(f"  With metadata (join hit):  {found}")
print(f"  Missing metadata (NaN):    {20-found}")

# Show first hit
for _, r in top20.iterrows():
    post = r["post"]
    if post in meta_ids:
        row = meta[meta["root_id"] == post].iloc[0]
        print(f"\n  First hit downstream: {post}")
        print(f"    synapse count = {r['count']}")
        print(f"    cell_type     = {row.get('cell_type', 'N/A')}")
        print(f"    region        = {row.get('region', 'N/A')}")
        print(f"    nt_pred       = {row.get('neurotransmitter_predicted', 'N/A')}")
        break

# Self-loop check
self_loops = edges[edges["pre"] == edges["post"]]
print(f"\nSelf-loops in edgelist: {len(self_loops)} ({len(self_loops)/len(edges)*100:.4f}%)")

# Overall coverage
all_neurons = set(edges["pre"]) | set(edges["post"])
covered = all_neurons & meta_ids
print(f"\nTotal unique neurons in edgelist: {len(all_neurons)}")
print(f"Covered by meta: {len(covered)} ({len(covered)/len(all_neurons)*100:.1f}%)")

print("\n" + "=" * 70)
print("2. NEURAL DYNAMICS — Making the connectome RUN")
print("=" * 70)

# Build a sub-circuit: the top_pre neuron + its top 50 downstream partners
N = 50
sub = (
    edges[edges["pre"] == top_pre]
    .sort_values("count", ascending=False)
    .head(N)
    .reset_index(drop=True)
)
posts = sub["post"].values.astype(np.int64)
counts = sub["count"].values.astype(np.float64)

# Adjacency: row 0 is the pre neuron, rows 1..N are downstream
# We only have pre->post connections here; make symmetric-ish local circuit
W = np.zeros((N + 1, N + 1))
W[0, 1:] = counts  # pre -> posts

# For posts that also connect among themselves (use edgelist to find)
posts_set = set(posts)
inter_edges = edges[
    edges["pre"].isin(posts_set) & edges["post"].isin(posts_set)
].copy()
pre_idx_map = {nid: i + 1 for i, nid in enumerate(posts)}
for _, r in inter_edges.iterrows():
    i = pre_idx_map.get(r["pre"])
    j = pre_idx_map.get(r["post"])
    if i is not None and j is not None:
        W[i, j] += r["count"]

# Normalize weights to max 1.0 for stability
W = W / (W.max() + 1e-9)

# Simple leaky integrate-and-fire (LIF)
np.random.seed(0)
dt = 0.1  # ms
T = 500   # ms total
n_steps = int(T / dt)
tau = 20.0  # membrane time constant ms
v_rest = -65.0
v_thresh = -55.0
v_reset = -70.0
refractory = 5.0  # ms

V = np.full(N + 1, v_rest)
refrac = np.zeros(N + 1)
spikes = np.zeros((n_steps, N + 1), dtype=bool)

# Drive: step current into pre neuron (index 0) after 50 ms
I_ext = np.zeros(N + 1)
for t_idx in range(n_steps):
    t = t_idx * dt
    if 50 <= t <= 200:
        I_ext[0] = 15.0  # nA, strong drive to pre
    elif 250 <= t <= 400:
        I_ext[0] = 8.0
    else:
        I_ext[0] = 0.0

    # Synaptic input from spikes at previous step
    # Use simple instantaneous synapses for demo
    spike_vec = spikes[max(0, t_idx - 1)] if t_idx > 0 else np.zeros(N + 1, dtype=bool)
    I_syn = W.T @ spike_vec.astype(float) * 20.0  # synaptic gain

    # Euler step
    dV = (-(V - v_rest) + I_ext + I_syn) * (dt / tau)
    V = V + dV

    # Refractory
    refrac = np.maximum(refrac - dt, 0)
    can_fire = refrac <= 0

    # Spike
    fired = (V >= v_thresh) & can_fire
    spikes[t_idx] = fired
    V[fired] = v_reset
    refrac[fired] = refractory

# Report
spike_counts = spikes.sum(axis=0)
print(f"\nSimulated {N+1} neurons for {T} ms ({n_steps} steps)")
print(f"  Pre neuron (id={top_pre}) fired {spike_counts[0]} times")
print(f"  Firing rates (Hz):")
for i in range(N + 1):
    nid = top_pre if i == 0 else int(posts[i - 1])
    rate = spike_counts[i] / (T / 1000)
    label = "PRE" if i == 0 else f"POST_{i}"
    print(f"    [{label}] id={nid}: {spike_counts[i]} spikes = {rate:.1f} Hz")

# Show which downstream neurons got metadata
print(f"\n  Downstream neurons with metadata among these {N}:")
for i in range(1, N + 1):
    nid = int(posts[i - 1])
    if nid in meta_ids:
        row = meta[meta["root_id"] == nid].iloc[0]
        ct = row.get("cell_type", "N/A")
        reg = row.get("region", "N/A")
        nt = row.get("neurotransmitter_predicted", "N/A")
        rate = spike_counts[i] / (T / 1000)
        if spike_counts[i] > 0:
            print(f"    id={nid} | {rate:.1f} Hz | type={ct} | region={reg} | nt={nt}")

print("\n" + "=" * 70)
print("✅ BANC connectome is now RUNNING in a simple LIF simulation.")
print("=" * 70)
