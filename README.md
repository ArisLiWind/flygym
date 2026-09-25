# FlyGym + BANC v888 — Neural Connectome to Biomechanics Bridge

> **Original**: [NeLy-EPFL/flygym](https://github.com/NeLy-EPFL/flygym) — FlyGym 2.1.0, a MuJoCo-based biomechanical simulation of *Drosophila melanogaster*.
>
> **This fork** adds a direct bridge from the **BANC v888** real neural connectome (Harvard Dataverse) into FlyGym's motor control loop, demonstrating a closed-loop pipeline: **real neuron spikes → CPG modulation → leg actuators → walking simulation**.

---

## What We Added

### 1. BANC Data Loading & Verification

| Script | Purpose |
|---|---|
| `scripts/test_banc.py` | Minimal check: loads `meta.feather` + `edgelist.feather`, prints shapes, runs a join |
| `scripts/test_banc_real.py` | Honest validation: overlap coverage, self-loop check, LIF spiking demo |

**Download** (run once):
```bash
cd data/banc
curl -L -o banc_888_meta.feather "https://dataverse.harvard.edu/api/access/datafile/14033740"
curl -L -o banc_888_edgelist_simple_v2.feather "https://dataverse.harvard.edu/api/access/datafile/13992792"
```

### 2. BANC Neural Dynamics

| Script | Purpose |
|---|---|
| `scripts/run_banc_lif.py` | LIF spiking network of a real BANC descending neuron + 50 downstream partners |
| `scripts/run_banc_network.py` | 200-neuron random recurrent subgraph (scalable network) |

Both scripts output a `banc_*_raster.csv` for spike-time analysis.

### 3. BANC → FlyGym Bridge ⭐

| Script | Purpose |
|---|---|
| `scripts/banc_flygym_bridge.py` | **End-to-end bridge**: BANC neuron spike rate → CPG amplitude modulation → NeuroMechFly walking |

**How it works**:
1. Loads BANC connectome, selects a **descending neuron** (e.g. `DNpe053`, dopaminergic)
2. Runs a **LIF simulation** on its local circuit to compute a realistic firing rate
3. Maps that rate to a **CPG amplitude gain** (e.g. 96 Hz → 1.60×)
4. Runs FlyGym's **tripod CPG controller** with the modulated gain
5. Outputs a rendered video of the biomechanical walking

**Run it**:
```bash
uv run python scripts/banc_flygym_bridge.py
# → outputs scripts/banc_bridge_output/banc_bridge_walk.mp4
```

### 4. Closed-Loop Sensory-Neural-Motor Integration 🔁

| Script | Purpose |
|---|---|
| `scripts/banc_flygym_closed_loop.py` | **Real closed loop**: FlyGym sensors → BANC sensory neurons → LIF + STDP → motor neurons → leg actuators |
| `scripts/build_banc_subnet.py` | Pre-builds a fast-loading `.npz` subnet from BANC feather files |

**How it works**:
1. Extracts a **leg-relevant subgraph** from BANC v888:
   - **18 sensory neurons** (tactile / proprioception on legs)
   - **20 interneurons** (connect sensory and motor pools)
   - **12 motor neurons** (`leg_motor` / `body_part_effector=leg`)
2. At **every physics step** (0.1 ms):
   - Reads **joint angles** from FlyGym as proprioceptive sensors
   - Encodes them as input currents to the sensory population
   - Advances the **LIF network** with real BANC synaptic weights
   - Applies **STDP** online (`A_pre=A_post=0.01`, `tau_trace=20 ms`)
   - Reads motor-pool spikes and converts them to **leg-specific CPG gains**
   - Drives NeuroMechFly leg actuators
3. The network **adapts** its weights while the fly walks

**Run it** (pre-build subnet first):
```bash
.venv/bin/python scripts/build_banc_subnet.py   # one-time setup
.venv/bin/python scripts/banc_flygym_closed_loop.py
# → outputs scripts/closed_loop_output/closed_loop.mp4
```

**Result**: 50 neurons, 5000 physics steps, **~974 total spikes**, real-time closed-loop walking simulation.
```

---

## Quick Start (Full Setup)

```bash
# 1. Clone this repo
git clone https://github.com/ArisLiWind/flygym.git
cd flygym

# 2. Install FlyGym + dependencies (Python 3.14, managed by uv)
uv sync --extra examples

# 3. Download BANC v888 data (345 MB total)
mkdir -p data/banc
curl -L -o data/banc/banc_888_meta.feather "https://dataverse.harvard.edu/api/access/datafile/14033740"
curl -L -o data/banc/banc_888_edgelist_simple_v2.feather "https://dataverse.harvard.edu/api/access/datafile/13992792"

# 4. Run the bridge
uv run python scripts/banc_flygym_bridge.py
```

---

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  BANC Connectome│     │   LIF Spiking   │     │   CPG Controller │
│  (188K neurons) │ ──▶ │   Network       │ ──▶ │   (modulated)    │
│  11.7M synapses │     │  (rate → gain)  │     │                  │
└─────────────────┘     └─────────────────┘     └────────┬─────────┘
                                                           │
                                                           ▼
                                              ┌─────────────────────┐
                                              │  NeuroMechFly       │
                                              │  MuJoCo Physics     │
                                              │  Position Actuators │
                                              └─────────────────────┘
```

### Open-loop bridge (banc_flygym_bridge.py)

```
┌─────────────────┐     ┌─────────────────┐     ┌──────────────────┐
│  BANC Connectome│     │   LIF Spiking   │     │   CPG Controller │
│  (188K neurons) │ ──▶ │   Network       │ ──▶ │   (modulated)    │
│  11.7M synapses │     │  (rate → gain)  │     │                  │
└─────────────────┘     └─────────────────┘     └────────┬─────────┘
                                                           │
                                                           ▼
                                              ┌─────────────────────┐
                                              │  NeuroMechFly       │
                                              │  MuJoCo Physics     │
                                              │  Position Actuators │
                                              └─────────────────────┘
```

**Key insight**: BANC provides the *static* connectivity. LIF turns it into *dynamic* spike trains. The spike rate of a descending neuron is interpreted as a **motor command intensity**, which scales the CPG's intrinsic amplitude.

### Closed-loop integration (banc_flygym_closed_loop.py)

```
        ┌─────────────────────────────────────────────────────────────┐
        │                         BANC v888 subnet                    │
        │  ┌──────────┐    ┌────────────┐    ┌──────────────┐        │
        │  │ Sensory  │───▶│ Interneuron│───▶│   Motor      │        │
        │  │ (18)     │    │ (20)       │    │ (12)         │        │
        │  └──────────┘    └────────────┘    └──────────────┘        │
        │       ▲                                    │                │
        │       │                                    │                │
        │   proprioception                         leg gains          │
        │   (joint angles)                            │                │
        └─────────────────────────────────────────────────────────────┘
                          │                              │
              ┌───────────┘                              └───────────┐
              │                                                      │
              ▼                                                      ▼
    ┌──────────────────┐                              ┌──────────────────┐
    │  FlyGym sensors  │                              │  FlyGym actuators│
    │  (read qpos)     │                              │  (CPG+gain)      │
    └──────────────────┘                              └──────────────────┘
              │                                              │
              └──────────────────────────────────────────────┘
                              NeuroMechFly physics
```

**Key insight**: This is a *true closed loop*. Every 0.1 ms physics step:
1. Joint angles are read from the biomechanical model
2. BANC sensory neurons are driven by proprioceptive input
3. The spiking network (LIF + STDP) computes the next motor command
4. Motor-neuron firing rates set per-leg CPG gains
5. The fly's legs move, changing joint angles → back to step 1

---

## What Was Changed vs. Original FlyGym

| File | Change | Why |
|---|---|---|
| `scripts/launch_interactive_viewer.py` | `Fly` → `NeuroMechFly` | API deprecation fix |
| `pyproject.toml` | `+pyarrow>=25.0.1` | Read `.feather` files |
| `.gitignore` | `+data/`, `+*raster.csv` | Exclude large data & outputs |
| `scripts/test_banc.py` | **new** | Data verification |
| `scripts/test_banc_real.py` | **new** | Coverage + LIF demo |
| `scripts/run_banc_lif.py` | **new** | Main neural dynamics |
| `scripts/run_banc_network.py` | **new** | Recurrent subgraph |
| `scripts/banc_flygym_bridge.py` | **new** | **Open-loop bridge** |
| `scripts/banc_flygym_closed_loop.py` | **new** | **Closed-loop sensory-neural-motor** |
| `scripts/build_banc_subnet.py` | **new** | Pre-build BANC subnet for fast loading |

**No FlyGym core code was modified** (`src/flygym/` untouched).

---

## Future Directions

- **Map specific cell types** (motor neurons, sensory neurons) to specific leg actuators using known anatomical projections
- **Add synaptic plasticity** (STDP) to the LIF layer so the bridge learns from simulation feedback
- **Replace CPG with full BANC recurrent network** — instead of modulating CPG, directly map a BANC subgraph's activity to all 42 actuators
- **Add sensory feedback** — use FlyGym's joint angles / contact forces to drive BANC sensory neurons in a closed loop

---

## Citation

- **FlyGym**: Wang-Chen et al., *Nature Methods* (2024) — https://neuromechfly.org
- **BANC v888**: Published on Harvard Dataverse, DOI: `10.7910/DVN/7WTH1N`

---

Maintained by: ArisLiWind / AzeleaLee
