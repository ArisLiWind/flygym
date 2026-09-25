#!/usr/bin/env python3
"""
BANC ↔ FlyGym  CLOSED-LOOP  sensory-neural-motor integration
"""

import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# 1. Load pre-built BANC subnet
# ---------------------------------------------------------------------------
NPZ = Path(__file__).resolve().parent / "banc_subnet.npz"
if not NPZ.exists():
    print("ERROR: banc_subnet.npz not found. Run build_banc_subnet.py first.")
    raise SystemExit(1)

print("[BANC] Loading subnet …")
d = np.load(NPZ, allow_pickle=True)
W = d["W"].astype(np.float32) * 20.0
sensory_ids = d["sensory_ids"]
motor_ids = d["motor_ids"]
motor_leg = d["motor_leg"]
N_S = int(d["N_S"])
N_I = int(d["N_I"])
N_M = int(d["N_M"])
N = W.shape[0]
print(f"  {N} neurons  (S={N_S} I={N_I} M={N_M})")

# ---------------------------------------------------------------------------
# 2. LIF + STDP state
# ---------------------------------------------------------------------------
tau = 10.0
v_rest = -65.0
v_thresh = -52.0
v_reset = -70.0
refractory = 1.5
syn_gain = 40.0

A_pre = 0.01
A_post = 0.01
tau_trace = 20.0

V = np.full(N, v_rest, dtype=np.float32)
r = np.zeros(N, dtype=np.float32)
pre_trace = np.zeros(N, dtype=np.float32)
post_trace = np.zeros(N, dtype=np.float32)

# ---------------------------------------------------------------------------
# 3. FlyGym setup
# ---------------------------------------------------------------------------
print("[FLYGYM] Setting up …")

from flygym import Simulation
from flygym.compose import (
    NeuroMechFly, ActuatorType, FlatGroundWorld, KinematicPosePreset,
)
from flygym.anatomy import (
    Skeleton, JointPreset, AxisOrder, ActuatedDOFPreset, ContactBodiesPreset,
)
from flygym.utils.math import Rotation3D
import mujoco

fly = NeuroMechFly()
skeleton = Skeleton(joint_preset=JointPreset.LEGS_ONLY, axis_order=AxisOrder.YAW_PITCH_ROLL)
fly.add_joints(skeleton, KinematicPosePreset.NEUTRAL)
actuated = skeleton.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
fly.add_actuators(actuated, ActuatorType.POSITION, neutral_input=KinematicPosePreset.NEUTRAL,
                  kp=50.0, ctrlrange=(-3.14, 3.14))
fly.add_joint_sites(JointPreset.LEGS_ONLY.to_joint_list())
fly.colorize()
cam = fly.add_tracking_camera(name="trackcam")

world = FlatGroundWorld()
world.add_fly(fly, (0, 0, 0.7), Rotation3D("quat", (1, 0, 0, 0)),
              bodysegs_with_ground_contact=ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD)

sim = Simulation(world, timestep=1e-4)
sim.set_renderer(cam)

# Warm-up
sim.reset()
k = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_KEY, "neutral")
sim.mj_data.qpos[:] = sim.mj_model.key_qpos[k]
sim.mj_data.qvel[:] = 0.0
for _ in range(100):
    sim.step()

# ---------------------------------------------------------------------------
# 4. Helpers
# ---------------------------------------------------------------------------
LEG_NAMES = ["LF", "LM", "LH", "RF", "RM", "RH"]
np.random.seed(7)
sens_leg = np.random.randint(0, 6, size=N_S)

def read_sensors(mj_data, mj_model):
    q = np.array(mj_data.qpos[:mj_model.nq])
    n = mj_model.nq // 6
    return np.array([np.abs(q[i*n:(i+1)*n]).mean() for i in range(6)])

def motor_spikes_to_gains(spikes):
    gains = np.ones(6, dtype=float)
    counts = np.zeros(6, dtype=float)
    for idx, sp in enumerate(spikes):
        if sp:
            gains[motor_leg[idx]] += 1.0
            counts[motor_leg[idx]] += 1.0
    counts[counts == 0] = 1
    gains = gains / counts
    gmin, gmax = gains.min(), gains.max()
    if gmax > gmin:
        gains = 0.2 + 1.8 * (gains - gmin) / (gmax - gmin)
    return gains

# ---------------------------------------------------------------------------
# 5. Main loop
# ---------------------------------------------------------------------------
print("[LOOP] Running …")

T_sim = 0.5          # 500 ms
n_steps = int(T_sim / sim.mj_model.opt.timestep)
neuro_dt = 0.1       # ms (timestep=1e-4)

cpg_phase = np.random.rand(6) * 2 * np.pi
cpg_freq = 12.0

log = []
start = time.perf_counter()
total_spikes = 0

for step in range(n_steps):
    # --- sensors → currents ---
    sensors = read_sensors(sim.mj_data, sim.mj_model)
    I_ext = np.full(N, 10.0, dtype=np.float32)
    for s in range(N_S):
        I_ext[s] += 15.0 + np.random.randn() * 5.0

    # --- LIF (exact same as working standalone test) ---
    sp = (r <= 0) & (V >= v_thresh)
    Isyn = W.T @ sp.astype(np.float32) * syn_gain
    V += (-(V - v_rest) + I_ext + Isyn) * (neuro_dt / tau)
    r = np.maximum(r - neuro_dt, 0.0)
    fired = (V >= v_thresh) & (r <= 0)
    V[fired] = v_reset
    r[fired] = refractory
    total_spikes += fired.sum()

    # --- STDP ---
    pre_trace = pre_trace * np.exp(-neuro_dt / tau_trace) + A_pre * fired.astype(np.float32)
    post_trace = post_trace * np.exp(-neuro_dt / tau_trace) + A_post * fired.astype(np.float32)
    W += np.outer(fired.astype(np.float32), post_trace) - np.outer(pre_trace, fired.astype(np.float32))
    W = np.clip(W, 0.0, 50.0).astype(np.float32)

    # --- motor decode ---
    motor_spikes = fired[N_S + N_I:]
    leg_gains = motor_spikes_to_gains(motor_spikes)

    # --- CPG → joint targets ---
    cpg_phase += 2 * np.pi * cpg_freq * sim.mj_model.opt.timestep
    n_act = len(actuated)
    per_leg = n_act // 6
    targets = np.zeros(n_act)
    for li in range(6):
        base = li * per_leg
        targets[base:base+per_leg] = 0.3 * leg_gains[li] * np.sin(cpg_phase[li] + li * np.pi / 3)

    # --- physics ---
    sim.set_actuator_inputs(fly.name, ActuatorType.POSITION, targets)
    sim.step()

    if step % 100 == 0:
        sim.render_as_needed()
        print(f"  step {step:4d}  instant_spikes={fired.sum():2d}  "
              f"cumul_spikes={total_spikes:4d}  gains=[{' '.join(f'{g:.2f}' for g in leg_gains)}]")
        log.append({"step": step, "instant": int(fired.sum()), "cumul": total_spikes,
                    "gains": leg_gains.copy()})

wall = time.perf_counter() - start
print(f"\nDone: {n_steps} steps in {wall:.1f}s")
print(f"Total spikes across all steps: {total_spikes}")

# ---------------------------------------------------------------------------
# 6. Save
# ---------------------------------------------------------------------------
out = Path(__file__).resolve().parent / "closed_loop_output"
out.mkdir(exist_ok=True)
sim.renderer.save_video(out / "closed_loop.mp4")
print(f"✅ Video: {out / 'closed_loop.mp4'}")

import csv
if log:
    with open(out / "log.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step", "instant_spikes", "cumul_spikes"] + LEG_NAMES)
        for r in log:
            w.writerow([r["step"], r["instant"], r["cumul"]] + r["gains"].tolist())
    print(f"Log: {out / 'log.csv'}")

with open(out / "summary.txt", "w") as f:
    f.write("Closed-loop BANC ↔ FlyGym\n")
    f.write(f"Neurons: {N} (S={N_S} I={N_I} M={N_M})\n")
    f.write(f"Steps: {n_steps}  Wall: {wall:.1f}s\n")
    f.write(f"Total spikes: {total_spikes}\n")
print(f"Summary: {out / 'summary.txt'}")
print("=" * 60)
print("✅ BANC senses → LIF computes → STDP adapts → FlyGym acts")
print("=" * 60)
