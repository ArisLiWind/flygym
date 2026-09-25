#!/usr/bin/env python3
"""BANC recurrent subgraph → running spiking network with many active neurons."""

import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "banc"
meta = pd.read_feather(DATA / "banc_888_meta.feather")
edges = pd.read_feather(DATA / "banc_888_edgelist_simple_v2.feather")

# Use neurons present in both tables
overlap = list(set(meta["root_id"]) & set(edges["pre"]) & set(edges["post"]))
print(f"Neurons in both meta and edgelist: {len(overlap)}")

# Sample 200 for a local recurrent circuit
np.random.seed(42)
subset = np.random.choice(overlap, size=min(200, len(overlap)), replace=False)
subset_set = set(subset)

# Extract sub-edges
sub_edges = edges[
    edges["pre"].isin(subset_set) & edges["post"].isin(subset_set)
].copy()
print(f"Edges within subset: {len(sub_edges)}")

# Index mapping
id2idx = {nid: i for i, nid in enumerate(subset)}
N = len(subset)

# Build sparse weight matrix
W = np.zeros((N, N))
for _, r in sub_edges.iterrows():
    i = id2idx[str(r["pre"])]
    j = id2idx[str(r["post"])]
    # weight proportional to synapse count, with clipping
    w = min(float(r["count"]) / 5.0, 80.0)
    W[i, j] += w

# LIF network parameters
dt = 0.1      # ms
T = 2000      # ms
n_steps = int(T / dt)
tau = 20.0    # ms
v_rest = -65.0
v_thresh = -52.0
v_reset = -70.0
refractory = 2.0

# Background Poisson input: each neuron gets ~8 Hz external spikes
ext_rate_hz = 8.0
ext_weight = 25.0
np.random.seed(7)

V = np.full(N, v_rest, dtype=float)
refrac = np.zeros(N)
spikes = np.zeros((n_steps, N), dtype=bool)
spike_times = [[] for _ in range(N)]

syn_gain = 30.0

for t_idx in range(n_steps):
    # Background Poisson spikes
    ext_spikes = np.random.rand(N) < (ext_rate_hz * dt / 1000.0)

    # Recurrent synaptic current from last step
    if t_idx > 0:
        I_rec = W.T @ spikes[t_idx - 1].astype(float) * syn_gain
    else:
        I_rec = np.zeros(N)

    I_ext = ext_spikes.astype(float) * ext_weight

    dV = (-(V - v_rest) + I_ext + I_rec) * (dt / tau)
    V = V + dV
    refrac = np.maximum(refrac - dt, 0)
    can_fire = refrac <= 0
    fired = (V >= v_thresh) & can_fire
    spikes[t_idx] = fired
    V[fired] = v_reset
    refrac[fired] = refractory

    for i in np.where(fired)[0]:
        spike_times[i].append(t_idx * dt)

# Results
rates = np.array([len(st) for st in spike_times]) / (T / 1000.0)
active = np.sum(rates > 0)
print(f"\n{'='*70}")
print(f"NETWORK SIMULATION: {N} neurons, {T} ms")
print(f"{'='*70}")
print(f"Active neurons (rate > 0 Hz): {active} / {N}")
print(f"Mean firing rate (all): {rates.mean():.2f} Hz")
print(f"Mean firing rate (active): {rates[rates>0].mean():.2f} Hz")
print(f"Max firing rate: {rates.max():.1f} Hz")

# Show top 10 most active neurons with metadata
print(f"\nTop 10 most active neurons:")
top_idx = np.argsort(rates)[::-1][:10]
for rank, i in enumerate(top_idx, 1):
    nid = str(subset[i])
    r = rates[i]
    row = meta[meta["root_id"] == nid]
    if len(row) > 0:
        ct = row.iloc[0].get("cell_type", "?")
        reg = row.iloc[0].get("region", "?")
        nt = row.iloc[0].get("neurotransmitter_predicted", "?")
    else:
        ct = reg = nt = "?"
    bar = "█" * int(r / 3)
    print(f"  #{rank} id={nid} | {r:5.1f} Hz | type={ct} | region={reg} | nt={nt} {bar}")

# Save raster
raster_path = Path(__file__).resolve().parent / "banc_network_raster.csv"
with open(raster_path, "w") as f:
    f.write("time_ms,neuron_idx,neuron_id,rate_hz,cell_type,region\n")
    for i, st in enumerate(spike_times):
        nid = str(subset[i])
        row = meta[meta["root_id"] == nid]
        ct = row.iloc[0].get("cell_type", "?") if len(row) > 0 else "?"
        reg = row.iloc[0].get("region", "?") if len(row) > 0 else "?"
        for t in st:
            f.write(f"{t:.1f},{i},{nid},{rates[i]:.2f},{ct},{reg}\n")
print(f"\nRaster saved to: {raster_path}")
print(f"{'='*70}")
print("✅ BANC connectome is now running as a RECURRENT spiking neural network.")
print(f"{'='*70}")
