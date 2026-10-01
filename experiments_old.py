"""
experiments.py - runs the experiments for the TPUv4 availability simulator.

Usage:
    python3 experiments.py          # run all experiments
    python3 experiments.py 1 3      # run only experiments 1 and 3

Each experiment prints its hypothesis, a results table, and saves
results/expN.csv and results/expN.png.

Depends on:
    pod.py         (Pod with randomize() and free_mask())
    allocators.py  (static_allocate, reconfigurable_allocate)
"""
import csv
import itertools
import os
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")              # save PNG files; no display needed
import matplotlib.pyplot as plt

from pod import Pod, State
from allocators import static_allocate, reconfigurable_allocate

# ---------------------------------------------------------------------------
# Global settings (state these assumptions in your report)
# ---------------------------------------------------------------------------
RESULTS_DIR = "results_old"
SEED = 42                     # fixed seed -> reproducible results
TRIALS = 2000                 # random pod states per data point
MACHINES_PER_CUBE = 16        # paper Sec 2.2: one cube = 16 TPU machines
BASE_MACHINE_AVAIL = 0.999    # 99.9% per machine (paper Sec 2.1)
POLICIES = ["static", "reconfigurable"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cube_fail_prob(machine_avail):
    """A cube is unusable if ANY of its 16 machines is down."""
    return 1 - machine_avail ** MACHINES_PER_CUBE


def job_can_run(free, shape, policy):
    """Can a job of this shape get cubes, given the free-cube mask?"""
    if policy == "static":
        return static_allocate(free, shape) is not None
    return reconfigurable_allocate(free, int(np.prod(shape))) is not None


def availability(shape, p_fail, p_used, trials=TRIALS,
                 pod_shape=(4, 4, 4), seed=SEED):
    """Fraction of random pod states in which the job can be placed.
    Both policies see the SAME pod state in each trial (fair comparison)."""
    rng = np.random.default_rng(seed)
    pod = Pod(pod_shape)
    ok = {p: 0 for p in POLICIES}
    for _ in range(trials):
        pod.randomize(p_fail, p_used, rng)
        free = pod.free_mask()
        for policy in POLICIES:
            ok[policy] += job_can_run(free, shape, policy)
    return {p: ok[p] / trials for p in POLICIES}


def save_csv(name, header, rows):
    path = os.path.join(RESULTS_DIR, f"{name}.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  saved {path}")


def save_plot(fig, name):
    path = os.path.join(RESULTS_DIR, f"{name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  saved {path}")


def banner(title, hypothesis):
    print("\n" + "=" * 70)
    print(title)
    print("-" * 70)
    print("HYPOTHESIS:", hypothesis)
    print("=" * 70)


# ---------------------------------------------------------------------------
# Background jobs (used by Experiment 3): occupancy that comes from OTHER
# jobs occupying blocks of cubes, instead of independent random cubes.
# ---------------------------------------------------------------------------
BACKGROUND_SHAPES = [(1, 1, 1), (1, 1, 2), (1, 2, 2), (2, 2, 2)]


def all_placements(free, shape):
    """Every position (x, y, z, a, b, c) where this shape fits on free cubes."""
    X, Y, Z = free.shape
    spots = []
    for a, b, c in set(itertools.permutations(shape)):
        if a > X or b > Y or c > Z:
            continue
        for x in range(X - a + 1):
            for y in range(Y - b + 1):
                for z in range(Z - c + 1):
                    if free[x:x+a, y:y+b, z:z+c].all():
                        spots.append((x, y, z, a, b, c))
    return spots


def fill_background(pod, target_occupancy, strategy, rng):
    """Place small background jobs until target_occupancy of the pod is USED.
    strategy = "random": each job lands at a random valid spot (scattered)
    strategy = "packed": each job lands at the lowest (x, y, z) spot (compact)
    """
    free = pod.free_mask()
    target = int(round(target_occupancy * free.size))
    used = 0
    while used < target:
        remaining = target - used
        options = [s for s in BACKGROUND_SHAPES if np.prod(s) <= remaining]
        shape = options[rng.integers(len(options))]
        spots = all_placements(free, shape)
        if not spots:
            spots = all_placements(free, (1, 1, 1))   # fall back to 1 cube
            if not spots:
                break                                  # pod is full
        if strategy == "random":
            x, y, z, a, b, c = spots[rng.integers(len(spots))]
        else:
            x, y, z, a, b, c = min(spots)              # corner-first
        free[x:x+a, y:y+b, z:z+c] = False
        for cube in pod.grid[x:x+a, y:y+b, z:z+c].flat:
            cube.state = State.USED
        used += a * b * c


# ---------------------------------------------------------------------------
# Experiment 1: job size (reproduces the shape of paper Figure 1)
# ---------------------------------------------------------------------------
def experiment_1():
    banner("EXPERIMENT 1 - Job size vs availability",
           "Static availability will drop sharply as the job grows, because "
           "it needs one contiguous all-healthy block. Reconfigurable will stay "
           "near 100% until the job approaches the number of healthy cubes.")

    shapes = [(1, 1, 1), (1, 1, 2), (1, 2, 2), (2, 2, 2), (2, 2, 4),
              (2, 4, 4), (3, 4, 4), (4, 4, 4)]
    p_fail = cube_fail_prob(BASE_MACHINE_AVAIL)
    occupancies = [0.0, 0.2]      # idle pod vs pod 20% used by other jobs
    print(f"  cube failure prob = {p_fail:.4f} (machine avail {BASE_MACHINE_AVAIL})")

    rows = []
    for p_used in occupancies:
        for shape in shapes:
            res = availability(shape, p_fail, p_used)
            n = int(np.prod(shape))
            rows.append([p_used, str(shape), n, res["static"], res["reconfigurable"]])
            print(f"  used={p_used:.1f} job={str(shape):10s} ({n:2d} cubes)  "
                  f"static={res['static']:.3f}  reconfig={res['reconfigurable']:.3f}")
    save_csv("exp1", ["occupancy", "shape", "cubes", "static", "reconfigurable"], rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for p_used, style in zip(occupancies, ["-", "--"]):
        sub = [r for r in rows if r[0] == p_used]
        sizes = [r[2] for r in sub]
        ax.plot(sizes, [r[3] * 100 for r in sub], "o" + style, color="tab:red",
                label=f"Static, {int(p_used*100)}% occupied")
        ax.plot(sizes, [r[4] * 100 for r in sub], "s" + style, color="tab:blue",
                label=f"Reconfigurable, {int(p_used*100)}% occupied")
    ax.set_xlabel("Job size (cubes; 1 cube = 64 TPUs)")
    ax.set_ylabel("Job availability (%)")
    ax.set_title("Exp 1: Availability vs job size")
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax.legend()
    save_plot(fig, "exp1")


# ---------------------------------------------------------------------------
# Experiment 2: per-machine availability (tests paper Sec 2.1 claim:
# reconfigurability lowers the per-host requirement from 99.9% to 99%)
# ---------------------------------------------------------------------------
def experiment_2():
    banner("EXPERIMENT 2 - Per-machine availability vs job availability",
           "A static pod will need very high per-machine availability (>=99.9%) "
           "for large jobs to run, while reconfigurable will tolerate ~99%.")

    machine_avails = [0.99, 0.995, 0.998, 0.999, 0.9995, 0.9999]
    shapes = [(2, 2, 4), (2, 4, 4)]      # 16-cube and 32-cube jobs
    rows = []
    for shape in shapes:
        for m in machine_avails:
            res = availability(shape, cube_fail_prob(m), p_used=0.0)
            rows.append([str(shape), m, cube_fail_prob(m),
                         res["static"], res["reconfigurable"]])
            print(f"  job={str(shape):10s} machine={m:.4f} "
                  f"cube_fail={cube_fail_prob(m):.4f}  "
                  f"static={res['static']:.3f}  reconfig={res['reconfigurable']:.3f}")
    save_csv("exp2", ["shape", "machine_avail", "cube_fail_prob",
                      "static", "reconfigurable"], rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = [f"{m*100:g}%" for m in machine_avails]
    for shape, style in zip(shapes, ["-", "--"]):
        sub = [r for r in rows if r[0] == str(shape)]
        n = int(np.prod(shape))
        ax.plot(labels, [r[3] * 100 for r in sub], "o" + style, color="tab:red",
                label=f"Static, {n}-cube job")
        ax.plot(labels, [r[4] * 100 for r in sub], "s" + style, color="tab:blue",
                label=f"Reconfigurable, {n}-cube job")
    ax.set_xlabel("Per-machine availability")
    ax.set_ylabel("Job availability (%)")
    ax.set_title("Exp 2: How reliable must each machine be?")
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax.legend()
    save_plot(fig, "exp2")


# ---------------------------------------------------------------------------
# Experiment 3 (NOT evaluated in the paper): fragmentation from co-tenant
# jobs. Same occupancy, different LAYOUT of the other jobs.
# ---------------------------------------------------------------------------
def experiment_3(trials=500):
    banner("EXPERIMENT 3 - Fragmentation from background jobs (not in paper)",
           "At the same occupancy, scattered (random) background jobs will hurt "
           "static availability much more than packed ones. Reconfigurable will "
           "not care about layout, only about the number of free cubes.")

    job_shape = (2, 2, 4)                     # 16-cube job
    occupancies = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    strategies = ["random", "packed"]
    p_fail = cube_fail_prob(BASE_MACHINE_AVAIL)

    rows = []
    for strategy in strategies:
        for occ in occupancies:
            rng = np.random.default_rng(SEED)
            pod = Pod((4, 4, 4))
            ok = {p: 0 for p in POLICIES}
            for _ in range(trials):
                pod.randomize(p_fail, 0.0, rng)      # failures only
                fill_background(pod, occ, strategy, rng)
                free = pod.free_mask()
                for policy in POLICIES:
                    ok[policy] += job_can_run(free, job_shape, policy)
            s, r = ok["static"] / trials, ok["reconfigurable"] / trials
            rows.append([strategy, occ, s, r])
            print(f"  layout={strategy:6s} occupancy={occ:.1f}  "
                  f"static={s:.3f}  reconfig={r:.3f}")
    save_csv("exp3", ["layout", "occupancy", "static", "reconfigurable"], rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for strategy, style in zip(strategies, ["-", "--"]):
        sub = [r for r in rows if r[0] == strategy]
        occ = [r[1] * 100 for r in sub]
        ax.plot(occ, [r[2] * 100 for r in sub], "o" + style, color="tab:red",
                label=f"Static, {strategy} layout")
        ax.plot(occ, [r[3] * 100 for r in sub], "s" + style, color="tab:blue",
                label=f"Reconfigurable, {strategy} layout")
    ax.set_xlabel("Pod occupied by other jobs (%)")
    ax.set_ylabel("Availability of a 16-cube job (%)")
    ax.set_title("Exp 3: Same occupancy, different fragmentation")
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax.legend()
    save_plot(fig, "exp3")


# ---------------------------------------------------------------------------
# Experiment 4 (optional, NOT in paper, useful for Part 4): scale the pod.
# The job is always half the pod.
# ---------------------------------------------------------------------------
def experiment_4(trials=500):
    banner("EXPERIMENT 4 - Growing the pod (job = half the pod, not in paper)",
           "Making the pod bigger will NOT keep static availability constant: "
           "a job that is half of a bigger pod has more cubes that must all be "
           "healthy at once, so static availability will fall with scale.")

    configs = [((2, 2, 2), (1, 2, 2)),
               ((4, 4, 4), (2, 4, 4)),
               ((6, 6, 6), (3, 6, 6)),
               ((8, 8, 8), (4, 8, 8))]
    p_fail = cube_fail_prob(BASE_MACHINE_AVAIL)
    rows = []
    for pod_shape, job_shape in configs:
        res = availability(job_shape, p_fail, 0.0, trials=trials, pod_shape=pod_shape)
        pod_cubes, job_cubes = int(np.prod(pod_shape)), int(np.prod(job_shape))
        rows.append([pod_cubes, job_cubes, pod_cubes * 64,
                     res["static"], res["reconfigurable"]])
        print(f"  pod={pod_cubes:3d} cubes ({pod_cubes*64:5d} TPUs) job={job_cubes:3d} cubes  "
              f"static={res['static']:.3f}  reconfig={res['reconfigurable']:.3f}")
    save_csv("exp4", ["pod_cubes", "job_cubes", "pod_tpus",
                      "static", "reconfigurable"], rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    tpus = [r[2] for r in rows]
    ax.plot(tpus, [r[3] * 100 for r in rows], "o-", color="tab:red", label="Static")
    ax.plot(tpus, [r[4] * 100 for r in rows], "s-", color="tab:blue",
            label="Reconfigurable")
    ax.set_xlabel("Pod size (TPUs)")
    ax.set_ylabel("Availability of a half-pod job (%)")
    ax.set_title("Exp 4: Bigger system, same relative job size")
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax.legend()
    save_plot(fig, "exp4")


# ---------------------------------------------------------------------------
EXPERIMENTS = {"1": experiment_1, "2": experiment_2,
               "3": experiment_3, "4": experiment_4}

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    chosen = sys.argv[1:] or list(EXPERIMENTS)
    for key in chosen:
        start = time.time()
        EXPERIMENTS[key]()
        print(f"  (took {time.time() - start:.1f}s)")