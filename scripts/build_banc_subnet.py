#!/usr/bin/env python3
"""Pre-build a small BANC leg subnet and save as .npz for fast loading."""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data" / "banc"
OUT = Path(__file__).resolve().parent / "banc_subnet.npz"

META = pd.read_feather(DATA / "banc_888_meta.feather")
EDGES = pd.read_feather(DATA / "banc_888_edgelist_simple_v2.feather")

META["root_id"] = META["root_id"].astype(str)
EDGES["pre"] = EDGES["pre"].astype(str)
EDGES["post"] = EDGES["post"].astype(str)

print("Building subnet …")

# Sensory
mask_s = (
    META["body_part_sensory"].isin(["front_leg", "middle_leg", "hind_leg"])
    | META["cell_function"].isin(["tactile", "proprioception"])
)
sensory_ids = sorted(list(set(META.loc[mask_s, "root_id"]) & set(EDGES["pre"])))[:18]
N_S = len(sensory_ids)

# Motor
mask_m = (META["cell_function"] == "leg_motor") | META["body_part_effector"].isin(
    ["front_leg", "middle_leg", "hind_leg"]
)
motor_ids = sorted(list(set(META.loc[mask_m, "root_id"]) & set(EDGES["post"])))[:12]
N_M = len(motor_ids)

# Interneurons
post_of_sens = set(EDGES.loc[EDGES["pre"].isin(sensory_ids), "post"])
pre_of_motor = set(EDGES.loc[EDGES["post"].isin(motor_ids), "pre"])
cand = list(post_of_sens & pre_of_motor - set(sensory_ids) - set(motor_ids))
np.random.seed(7)
intern_ids = sorted(np.random.choice(cand, size=min(20, len(cand)), replace=False)) if cand else []
N_I = len(intern_ids)

NEURON_IDS = sensory_ids + intern_ids + motor_ids
N = len(NEURON_IDS)
print(f"  {N} neurons  S={N_S} I={N_I} M={N_M}")

id2idx = {nid: i for i, nid in enumerate(NEURON_IDS)}

sub = EDGES[EDGES["pre"].isin(NEURON_IDS) & EDGES["post"].isin(NEURON_IDS)].copy()
sub["wi"] = sub["pre"].map(id2idx)
sub["wj"] = sub["post"].map(id2idx)
sub["w"] = sub["count"].astype(float) / 5.0

W = np.zeros((N, N), dtype=np.float32)
for _, r in sub.iterrows():
    W[int(r["wi"]), int(r["wj"])] += float(r["w"])
rs = W.sum(axis=1, keepdims=True)
W = np.where(rs > 0, W / rs.clip(min=1.0), W).astype(np.float32)

# Motor leg assignment
motor_leg = np.zeros(N_M, dtype=int)
for idx, nid in enumerate(motor_ids):
    row = META[META["root_id"] == nid]
    if len(row):
        bp = str(row.iloc[0].get("body_part_effector", ""))
        if bp == "front_leg":
            motor_leg[idx] = idx % 2
        elif bp == "middle_leg":
            motor_leg[idx] = 1 + idx % 2
        else:
            motor_leg[idx] = 2 + idx % 2
    else:
        motor_leg[idx] = idx % 6

# Save
np.savez(
    OUT,
    W=W,
    sensory_ids=np.array(sensory_ids),
    intern_ids=np.array(intern_ids),
    motor_ids=np.array(motor_ids),
    motor_leg=motor_leg,
    N_S=N_S,
    N_I=N_I,
    N_M=N_M,
)
print(f"Saved to {OUT}")
