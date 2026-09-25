#!/usr/bin/env python3
"""BANC connectome → real spiking neural network (LIF)."""

import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "banc"
meta = pd.read_feather(DATA / "banc_888_meta.feather")
edges = pd.read_feather(DATA / "banc_888_edgelist_simple_v2.feather")

# IDs are stored as strings in the feather files
meta_ids_set = set(meta["root_id"])

# Pick a well-documented pre neuron that actually appears in the edgelist
meta_has_out = meta.dropna(subset=["root_id", "output_connections"]).copy()
meta_has_out = meta_has_out[meta_has_out["output_connections"] > 500]

# Filter to those present in edgelist
pre_ids_set = set(edges["pre"])
candidates = meta_has_out[meta_has_out["root_id"].isin(pre_ids_set)]
print(f"Candidates in both meta and edgelist: {len(candidates)}")

# Prefer known cell types
chosen = None
for preferred in ["DN", "PN", "MN", "AN", "KC", "MBON", "OA", "DA"]:
    cand = candidates[candidates["cell_type"].astype(str).str.startswith(preferred, na=False)]
    if len(cand) > 0:
        chosen = cand.sort_values("output_connections", ascending=False).iloc[0]
        break
if chosen is None:
    chosen = candidates.sort_values("output_connections", ascending=False).iloc[0]

pre_id = str(chosen["root_id"])
print("=" * 70)
print("SELECTED STARTER NEURON (from meta + edgelist)")
print("=" * 70)
print(f"root_id          : {pre_id}")
print(f"cell_type        : {chosen.get('cell_type')}")
print(f"region           : {chosen.get('region')}")
print(f"super_class      : {chosen.get('super_class')}")
print(f"output_conn      : {chosen.get('output_connections')}")
print(f"input_conn       : {chosen.get('input_connections')}")
print(f"nt_predicted     : {chosen.get('neurotransmitter_predicted')}")

# Top 20 downstream partners
top20 = (
    edges[edges["pre"] == pre_id]
    .sort_values("count", ascending=False)
    .head(20)
    .reset_index(drop=True)
)

print("\n" + "=" * 70)
print("TOP 20 DOWNSTREAM PARTNERS (with metadata)")
print("=" * 70)
for _, r in top20.iterrows():
    post = str(r["post"])
    cnt = int(r["count"])
    if post in meta_ids_set:
        row = meta[meta["root_id"] == post].iloc[0]
        ct = row.get("cell_type", "N/A")
        reg = row.get("region", "N/A")
        nt = row.get("neurotransmitter_predicted", "N/A")
        print(f"  post={post} | synapses={cnt:4d} | type={ct} | region={reg} | nt={nt}")
    else:
        print(f"  post={post} | synapses={cnt:4d} | (no meta)")

# Build local circuit: pre + top 50 downstream
N = 50
sub = (
    edges[edges["pre"] == pre_id]
    .sort_values("count", ascending=False)
    .head(N)
    .reset_index(drop=True)
)
posts = sub["post"].values.astype(str)
counts = sub["count"].values.astype(float)

# Build weight matrix (pre = idx 0, posts = 1..N)
W = np.zeros((N + 1, N + 1))
weights = np.clip(counts / 10.0, 0.5, 50.0)
W[0, 1:] = weights

# Recurrent connections among posts
posts_set = set(posts)
inter = edges[edges["pre"].isin(posts_set) & edges["post"].isin(posts_set)].copy()
idx_map = {nid: i + 1 for i, nid in enumerate(posts)}
for _, r in inter.iterrows():
    i = idx_map.get(str(r["pre"]))
    j = idx_map.get(str(r["post"]))
    if i is not None and j is not None:
        W[i, j] += float(r["count"]) / 20.0

# LIF simulation
dt = 0.1
T = 1000
n_steps = int(T / dt)
tau = 15.0
v_rest = -65.0
v_thresh = -55.0
v_reset = -70.0
refractory = 3.0

V = np.full(N + 1, v_rest, dtype=float)
refrac = np.zeros(N + 1)
spikes = np.zeros((n_steps, N + 1), dtype=bool)
spike_times = [[] for _ in range(N + 1)]

I_ext = np.zeros(N + 1)
I_ext[0] = 22.0
I_ext[1:] = 8.0

syn_gain = 80.0

for t_idx in range(n_steps):
    t = t_idx * dt
    if t_idx > 0:
        spike_vec = spikes[t_idx - 1].astype(float)
        I_syn = W.T @ spike_vec * syn_gain
    else:
        I_syn = np.zeros(N + 1)

    dV = (-(V - v_rest) + I_ext + I_syn) * (dt / tau)
    V = V + dV
    refrac = np.maximum(refrac - dt, 0)
    can_fire = refrac <= 0
    fired = (V >= v_thresh) & can_fire
    spikes[t_idx] = fired
    V[fired] = v_reset
    refrac[fired] = refractory
    for i in np.where(fired)[0]:
        spike_times[i].append(t)

# Results
print("\n" + "=" * 70)
print("SPIKING RESULTS (Local Circuit Simulation)")
print("=" * 70)
print(f"Simulated {N+1} neurons for {T} ms")

total_spikes = [len(st) for st in spike_times]
for i in range(N + 1):
    nid = pre_id if i == 0 else str(posts[i - 1])
    rate = total_spikes[i] / (T / 1000.0)
    label = "STARTER" if i == 0 else f"POST_{i:02d}"
    meta_row = meta[meta["root_id"] == nid] if nid in meta_ids_set else None
    ct = meta_row.iloc[0].get("cell_type", "?") if meta_row is not None and len(meta_row) > 0 else "?"
    bar = "█" * int(rate / 2)
    print(f"  [{label}] id={nid} | {total_spikes[i]:3d} spikes | {rate:5.1f} Hz | type={ct} {bar}")

fired_downstream = [i for i in range(1, N + 1) if total_spikes[i] > 0]
print(f"\nDownstream neurons that fired: {len(fired_downstream)} / {N}")
if fired_downstream:
    print("Active downstream neurons:")
    for i in fired_downstream[:10]:
        nid = str(posts[i - 1])
        row = meta[meta["root_id"] == nid].iloc[0] if nid in meta_ids_set else None
        ct = row.get("cell_type", "?") if row is not None else "?"
        reg = row.get("region", "?") if row is not None else "?"
        nt = row.get("neurotransmitter_predicted", "?") if row is not None else "?"
        rate = total_spikes[i] / (T / 1000.0)
        print(f"  id={nid} | {rate:.1f} Hz | type={ct} | region={reg} | nt={nt}")

raster_path = Path(__file__).resolve().parent / "banc_lif_raster.csv"
with open(raster_path, "w") as f:
    f.write("time_ms,neuron_idx,neuron_id,label\n")
    for i, st in enumerate(spike_times):
        nid = pre_id if i == 0 else str(posts[i - 1])
        lab = "STARTER" if i == 0 else f"POST_{i:02d}"
        for t in st:
            f.write(f"{t:.1f},{i},{nid},{lab}\n")
print(f"\nRaster saved to: {raster_path}")
print("=" * 70)
print("✅ BANC connectome is now running as a real spiking neural network.")
print("=" * 70)
