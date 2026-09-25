#!/usr/bin/env python3
"""
BANC → FlyGym Bridge: Neural activity from real connectome drives biomechanical walking.

Proof-of-concept: A BANC descending neuron's spike rate modulates a CPG network,
which in turn drives NeuroMechFly leg actuators via position control.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# BANC: load connectome + simulate a local circuit with LIF
# ---------------------------------------------------------------------------
DATA = Path(__file__).resolve().parents[1] / "data" / "banc"
meta = pd.read_feather(DATA / "banc_888_meta.feather")
edges = pd.read_feather(DATA / "banc_888_edgelist_simple_v2.feather")

# Pick a descending neuron present in both meta and edgelist
pre_ids = set(edges["pre"])
meta_ids = set(meta["root_id"])
candidates = meta[
    meta["root_id"].isin(pre_ids)
    & meta["output_connections"].notna()
    & (meta["output_connections"] > 200)
    & meta["super_class"].astype(str).str.contains("descending", case=False, na=False)
]
if len(candidates) == 0:
    candidates = meta[meta["root_id"].isin(pre_ids) & meta["output_connections"].notna()]
chosen = candidates.sort_values("output_connections", ascending=False).iloc[0]

pre_id = str(chosen["root_id"])
print(f"[BANC] Starter neuron: {pre_id}")
print(f"  cell_type = {chosen.get('cell_type')}")
print(f"  region    = {chosen.get('region')}")
print(f"  nt_pred   = {chosen.get('neurotransmitter_predicted')}")
print(f"  outputs   = {chosen.get('output_connections')}")

# Build sub-circuit: pre + top 50 downstream
top50 = (
    edges[edges["pre"] == pre_id]
    .sort_values("count", ascending=False)
    .head(50)
    .reset_index(drop=True)
)
posts = top50["post"].values.astype(str)
counts = top50["count"].values.astype(float)

# Weight matrix (pre = 0, posts = 1..50)
W = np.zeros((51, 51))
W[0, 1:] = np.clip(counts / 5.0, 1.0, 80.0)

# Add recurrent edges among posts
posts_set = set(posts)
inter = edges[edges["pre"].isin(posts_set) & edges["post"].isin(posts_set)].copy()
idx_map = {nid: i + 1 for i, nid in enumerate(posts)}
for _, r in inter.iterrows():
    i = idx_map.get(str(r["pre"]))
    j = idx_map.get(str(r["post"]))
    if i is not None and j is not None:
        W[i, j] += float(r["count"]) / 10.0

# LIF parameters tuned for strong propagation
dt = 0.1       # ms
T_lif = 500    # ms
n_steps_lif = int(T_lif / dt)
tau = 15.0
v_rest = -65.0
v_thresh = -55.0
v_reset = -70.0
refractory = 2.0

V = np.full(51, v_rest, dtype=float)
refrac = np.zeros(51)
spikes = np.zeros((n_steps_lif, 51), dtype=bool)
spike_times = [[] for _ in range(51)]

I_ext = np.zeros(51)
I_ext[0] = 25.0   # strong tonic drive to descending neuron
I_ext[1:] = 4.0   # weak background

syn_gain = 60.0

for t_idx in range(n_steps_lif):
    if t_idx > 0:
        I_syn = W.T @ spikes[t_idx - 1].astype(float) * syn_gain
    else:
        I_syn = np.zeros(51)
    dV = (-(V - v_rest) + I_ext + I_syn) * (dt / tau)
    V = V + dV
    refrac = np.maximum(refrac - dt, 0)
    can_fire = refrac <= 0
    fired = (V >= v_thresh) & can_fire
    spikes[t_idx] = fired
    V[fired] = v_reset
    refrac[fired] = refractory
    for i in np.where(fired)[0]:
        spike_times[i].append(t_idx * dt)

# Compute descending modulation signal from pre neuron spike rate
pre_rate = len(spike_times[0]) / (T_lif / 1000.0)  # Hz
print(f"\n[BANC] Pre neuron firing rate: {pre_rate:.1f} Hz")

# ---------------------------------------------------------------------------
# FlyGym: CPG walking with descending modulation from BANC
# ---------------------------------------------------------------------------
print("\n[FLYGYM] Setting up NeuroMechFly + CPG controller...")

from flygym import Simulation
from flygym.compose import FlatGroundWorld
from flygym_demo.complex_terrain import (
    CPGController,
    LocomotionAction,
    PreprogrammedSteps,
    apply_locomotion_action,
    make_locomotion_fly,
    make_tripod_cpg_network,
)

fly = make_locomotion_fly(name="banc_bridge", add_adhesion=True, colorize=True)
cam = fly.add_tracking_camera(name="trackcam")

world = FlatGroundWorld()
world.add_fly(fly, [0, 0, 0.7], __import__("flygym.utils.math", fromlist=["Rotation3D"]).Rotation3D("quat", [1, 0, 0, 0]))

sim = Simulation(world, timestep=1e-4)
sim.set_renderer(cam)

preprogrammed_steps = PreprogrammedSteps()
dof_order = fly.get_actuated_jointdofs_order("position")

# Create CPG network
cpg_network = make_tripod_cpg_network(
    timestep=sim.timestep,
    intrinsic_amplitude=1.0,
    coupling_strength=10.0,
    convergence_coef=20.0,
    seed=0,
)
controller = CPGController(
    cpg_network=cpg_network,
    preprogrammed_steps=preprogrammed_steps,
    output_dof_order=dof_order,
)

# Descending modulation from BANC rate:
# map pre_rate [0, 150] Hz → amplitude multiplier [0.2, 2.0]
amplitude_gain = np.clip(pre_rate / 60.0, 0.3, 2.0)
print(f"  CPG amplitude gain from BANC: {amplitude_gain:.2f}x")

sim.reset()
initial_action = LocomotionAction(
    joint_angles=preprogrammed_steps.default_pose_by_dof_order(dof_order),
    adhesion_onoff=np.ones(6, dtype=bool),
)
apply_locomotion_action(sim, fly.name, initial_action)
sim.warmup()

run_time = 1.0  # 1 second of walking
nsteps_sim = int(run_time / sim.timestep)

print(f"[FLYGYM] Simulating {run_time}s walking ({nsteps_sim} steps)...")

for step_idx in range(nsteps_sim):
    # Modulate CPG amplitude with BANC-derived signal
    controller.cpg_network.intrinsic_amps = (
        np.ones(6) * amplitude_gain
    )
    action = controller.step()
    apply_locomotion_action(sim, fly.name, action)
    sim.step()
    sim.render_as_needed()

# Save video
out_dir = Path(__file__).resolve().parent / "banc_bridge_output"
out_dir.mkdir(exist_ok=True)
sim.renderer.save_video(out_dir / "banc_bridge_walk.mp4")
print(f"\n✅ Video saved: {out_dir / 'banc_bridge_walk.mp4'}")

# Also save a summary
with open(out_dir / "summary.txt", "w") as f:
    f.write(f"BANC starter neuron: {pre_id}\n")
    f.write(f"  cell_type: {chosen.get('cell_type')}\n")
    f.write(f"  nt:        {chosen.get('neurotransmitter_predicted')}\n")
    f.write(f"  rate:      {pre_rate:.1f} Hz\n")
    f.write(f"CPG amplitude gain: {amplitude_gain:.2f}x\n")
    f.write(f"Simulated: {run_time}s\n")
print(f"Summary saved: {out_dir / 'summary.txt'}")
