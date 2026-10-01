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
# Experiment 5 (NOT in paper, companion to Exp 4): scale the pod but keep
# the job FIXED at 32 cubes, so the extra cubes act as spares.
# Uses 99% machine availability so the static policy is under pressure.
# ---------------------------------------------------------------------------
def experiment_5(trials=500):
    banner("EXPERIMENT 5 - Growing the pod, FIXED 32-cube job (not in paper)",
           "When the job size stays fixed, extra cubes act as spares, so a "
           "bigger pod will raise static availability (more places to find a "
           "healthy block). Reconfigurable is already at 100%.")

    job_shape = (2, 4, 4)
    machine_avail = 0.99
    pods = [(4, 4, 4), (4, 4, 6), (4, 4, 8), (6, 6, 6), (8, 8, 8)]
    p_fail = cube_fail_prob(machine_avail)
    print(f"  machine avail {machine_avail} -> cube failure prob {p_fail:.4f}")

    rows = []
    for pod_shape in pods:
        res = availability(job_shape, p_fail, 0.0, trials=trials, pod_shape=pod_shape)
        pod_cubes = int(np.prod(pod_shape))
        rows.append([pod_cubes, pod_cubes * 64, res["static"], res["reconfigurable"]])
        print(f"  pod={pod_cubes:3d} cubes ({pod_cubes*64:5d} TPUs) job=32 cubes  "
              f"static={res['static']:.3f}  reconfig={res['reconfigurable']:.3f}")
    save_csv("exp5", ["pod_cubes", "pod_tpus", "static", "reconfigurable"], rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    tpus = [r[1] for r in rows]
    ax.plot(tpus, [r[2] * 100 for r in rows], "o-", color="tab:red", label="Static")
    ax.plot(tpus, [r[3] * 100 for r in rows], "s-", color="tab:blue",
            label="Reconfigurable")
    ax.set_xlabel("Pod size (TPUs)")
    ax.set_ylabel("Availability of a 32-cube job (%)")
    ax.set_title("Exp 5: Bigger system, fixed job size (extra cubes = spares)")
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax.legend()
    save_plot(fig, "exp5")


# ---------------------------------------------------------------------------
# Experiment 6 (production-driven): replay 100 weeks of Figure 12.
#
# The three lists below are the weekly average DAILY failure rates (in %)
# read from paper Figure 12 (a) machines, (b) ICI links, (c) OCS.
# They were digitized from the published figure, so they are approximate.
# Sanity check: their averages (0.077%, 0.0049%, 0.036%) match the averages
# stated in paper Sec 5.2 (0.08%, 0.005%, 0.04%).
# ---------------------------------------------------------------------------
FIG12_MACHINE = [
    0.0622, 0.0563, 0.0767, 0.0856, 0.0604, 0.0719, 0.1130, 0.1019, 0.0526, 0.0348,
    0.0430, 0.1044, 0.1174, 0.0937, 0.0885, 0.0500, 0.0659, 0.0785, 0.1104, 0.1170,
    0.1393, 0.1404, 0.0796, 0.0511, 0.0867, 0.0600, 0.0637, 0.0615, 0.0552, 0.0604,
    0.0656, 0.0656, 0.0481, 0.0485, 0.0733, 0.0804, 0.0674, 0.0670, 0.0689, 0.0552,
    0.0567, 0.0585, 0.0600, 0.0563, 0.0656, 0.0719, 0.0648, 0.1174, 0.1163, 0.1200,
    0.1041, 0.1141, 0.0922, 0.0737, 0.0796, 0.0759, 0.0663, 0.0633, 0.0633, 0.1115,
    0.1248, 0.0867, 0.0767, 0.0689, 0.0637, 0.0559, 0.0507, 0.0481, 0.1033, 0.0837,
    0.0544, 0.0681, 0.0789, 0.0519, 0.0496, 0.0622, 0.0759, 0.0770, 0.0944, 0.1070,
    0.1089, 0.1237, 0.1211, 0.1059, 0.0896, 0.0937, 0.0844, 0.0744, 0.0833, 0.0726,
    0.0681, 0.0915, 0.0626, 0.0485, 0.0589, 0.0656, 0.0685, 0.0578, 0.0548, 0.0607,
]
FIG12_ICI = [
    0.0044, 0.0032, 0.0044, 0.0044, 0.0054, 0.0057, 0.0053, 0.0042, 0.0046, 0.0032,
    0.0033, 0.0035, 0.0047, 0.0058, 0.0120, 0.0089, 0.0074, 0.0093, 0.0081, 0.0098,
    0.0070, 0.0061, 0.0060, 0.0059, 0.0040, 0.0036, 0.0037, 0.0040, 0.0034, 0.0031,
    0.0030, 0.0027, 0.0023, 0.0027, 0.0029, 0.0027, 0.0032, 0.0037, 0.0041, 0.0030,
    0.0051, 0.0041, 0.0040, 0.0036, 0.0035, 0.0041, 0.0040, 0.0043, 0.0044, 0.0043,
    0.0039, 0.0028, 0.0023, 0.0042, 0.0042, 0.0031, 0.0029, 0.0030, 0.0032, 0.0029,
    0.0031, 0.0034, 0.0031, 0.0040, 0.0064, 0.0060, 0.0050, 0.0041, 0.0046, 0.0044,
    0.0051, 0.0055, 0.0050, 0.0053, 0.0045, 0.0052, 0.0069, 0.0080, 0.0104, 0.0104,
    0.0089, 0.0089, 0.0084, 0.0063, 0.0047, 0.0049, 0.0065, 0.0041, 0.0048, 0.0050,
    0.0048, 0.0047, 0.0040, 0.0047, 0.0053, 0.0050, 0.0057, 0.0057, 0.0053, 0.0053,
]
FIG12_OCS = [
    0.0385, 0.0448, 0.0385, 0.0402, 0.0353, 0.0255, 0.0180, 0.0178, 0.0182, 0.0258,
    0.0228, 0.0302, 0.0305, 0.0330, 0.0298, 0.0245, 0.0238, 0.0195, 0.0282, 0.0245,
    0.0302, 0.0348, 0.0175, 0.0200, 0.0235, 0.0322, 0.0285, 0.0210, 0.0340, 0.0392,
    0.0322, 0.0295, 0.0370, 0.0427, 0.0620, 0.0545, 0.0505, 0.0560, 0.0555, 0.0560,
    0.0490, 0.0332, 0.0452, 0.0508, 0.0490, 0.0442, 0.0462, 0.0422, 0.0410, 0.0482,
    0.0410, 0.0338, 0.0420, 0.0510, 0.0370, 0.0360, 0.0575, 0.0622, 0.0590, 0.0580,
    0.0618, 0.0642, 0.0703, 0.0830, 0.0748, 0.0478, 0.0230, 0.0335, 0.0435, 0.0405,
    0.0462, 0.0425, 0.0472, 0.0438, 0.0405, 0.0400, 0.0422, 0.0328, 0.0162, 0.0108,
    0.0110, 0.0160, 0.0278, 0.0298, 0.0112, 0.0090, 0.0108, 0.0138, 0.0353, 0.0492,
    0.0185, 0.0145, 0.0142, 0.0205, 0.0318, 0.0260, 0.0152, 0.0245, 0.0335, 0.0230,
]

LINKS_PER_CUBE = 96     # paper Sec 2.2: 16 optical ICI links x 6 faces
NUM_OCS = 48            # paper Sec 2.2: 48 optical circuit switches per pod
REPAIR_DAYS = 1.0       # assumption: a failed part stays down ~1 day


def week_probabilities(week):
    """Turn Figure 12 daily failure rates (%) into 'down right now' probs."""
    p_machine = min(1.0, FIG12_MACHINE[week] / 100 * REPAIR_DAYS)
    p_link = min(1.0, FIG12_ICI[week] / 100 * REPAIR_DAYS)
    p_ocs = min(1.0, FIG12_OCS[week] / 100 * REPAIR_DAYS)
    # a cube is unusable if any of its 16 machines or 96 optical links is down
    p_cube = 1 - (1 - p_machine) ** MACHINES_PER_CUBE * (1 - p_link) ** LINKS_PER_CUBE
    return p_cube, p_ocs


def week_verdict(avail):
    if avail >= 0.95:
        return "GOOD"
    if avail >= 0.80:
        return "RISKY"
    return "POOR"


def experiment_6(trials=300, job_shape=(3, 4, 4), name="exp6"):
    banner("EXPERIMENT 6 - Replaying 100 weeks of real failure rates (Figure 12)",
           "Weeks with high machine/ICI failure rates will hurt the static pod "
           "the most. Reconfigurable without fault-tolerant (FT) routing will "
           "only lose weeks with OCS outages, and FT routing will recover most "
           "of those at a small performance cost.")

    n = int(np.prod(job_shape))
    print(f"  job = {job_shape} = {n} cubes ({n*64} TPUs), "
          f"repair time assumed {REPAIR_DAYS} day(s)")
    print(f"  policies: static | reconfig (no FT routing) | reconfig + FT routing")
    print(f"  verdict: GOOD >= 95%, RISKY 80-95%, POOR < 80%\n")
    print(f"  {'week':>4} {'mach%':>7} {'ici%':>7} {'ocs%':>6}   "
          f"{'static':>7} {'noFT':>7} {'FT':>7} {'degraded':>9}   verdicts (static/noFT/FT)")

    rng = np.random.default_rng(SEED)
    pod = Pod((4, 4, 4))
    rows = []
    for week in range(100):
        p_cube, p_ocs = week_probabilities(week)
        ok_static = ok_noft = ok_ft = degraded = 0
        for _ in range(trials):
            pod.randomize(p_cube, 0.0, rng)
            free = pod.free_mask()
            ocs_down = rng.binomial(NUM_OCS, p_ocs)

            # Static (TPUv3-style): fixed mesh, no OCS in the path
            ok_static += static_allocate(free, job_shape) is not None

            # Reconfigurable: needs n healthy cubes anywhere ...
            enough = reconfigurable_allocate(free, n) is not None
            # ... and without FT routing, every OCS must be up
            ok_noft += enough and ocs_down == 0
            # ... with FT routing, one OCS outage is tolerated (runs slower)
            if enough and ocs_down <= 1:
                ok_ft += 1
                degraded += ocs_down == 1

        s, nf, ft = ok_static / trials, ok_noft / trials, ok_ft / trials
        dg = degraded / trials
        verdicts = [week_verdict(v) for v in (s, nf, ft)]
        rows.append([week, FIG12_MACHINE[week], FIG12_ICI[week], FIG12_OCS[week],
                     round(p_cube, 5), s, nf, ft, dg] + verdicts)
        print(f"  {week:4d} {FIG12_MACHINE[week]:7.4f} {FIG12_ICI[week]:7.4f} "
              f"{FIG12_OCS[week]:6.3f}   {s:7.3f} {nf:7.3f} {ft:7.3f} {dg:9.3f}   "
              f"{'/'.join(verdicts)}")

    save_csv(name, ["week", "machine_rate_pct", "ici_rate_pct", "ocs_rate_pct",
                      "cube_fail_prob", "static", "reconfig_noFT", "reconfig_FT",
                      "degraded_FT", "verdict_static", "verdict_noFT",
                      "verdict_FT"], rows)

    # --- summary ---------------------------------------------------------
    print("\n  SUMMARY over 100 weeks")
    for idx, label in ((5, "static"), (6, "reconfig no FT"), (7, "reconfig + FT")):
        vals = [r[idx] for r in rows]
        v = [week_verdict(x) for x in vals]
        worst = int(np.argmin(vals))
        print(f"  {label:15s} mean={np.mean(vals):.3f}  worst={min(vals):.3f} "
              f"(week {worst})  GOOD={v.count('GOOD')} RISKY={v.count('RISKY')} "
              f"POOR={v.count('POOR')}")
    print(f"  FT routing ran degraded (1 OCS down) in "
          f"{np.mean([r[8] for r in rows])*100:.1f}% of trials on average")

    # --- plot ------------------------------------------------------------
    weeks = [r[0] for r in rows]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(weeks, [r[5] * 100 for r in rows], color="tab:red", label="Static")
    ax1.plot(weeks, [r[6] * 100 for r in rows], color="tab:blue",
             label="Reconfigurable, no FT routing")
    ax1.plot(weeks, [r[7] * 100 for r in rows], color="tab:orange",
             label="Reconfigurable + FT routing")
    ax1.axhline(95, color="gray", ls=":", lw=1)
    ax1.set_ylabel(f"Availability of {n}-cube job (%)")
    ax1.set_title(f"Exp 6: Weekly availability of a {n}-cube job (Figure 12 failure rates)")
    ax1.set_ylim(-3, 103)
    ax1.grid(alpha=0.3)
    ax1.legend(loc="lower left", fontsize=8)

    ax2.plot(weeks, FIG12_MACHINE, color="tab:red", label="Machine")
    ax2.plot(weeks, FIG12_OCS, color="tab:green", label="OCS")
    ax2.set_ylabel("Daily failure rate (%)")
    ax2.set_xlabel("Week")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="upper right", fontsize=8)
    save_plot(fig, name)


# ---------------------------------------------------------------------------
# Experiment 7 (production-driven): replay 60 days of Figure 13.
#
# Figure 13 gives, per day, the jobs admitted (blue) and the OCS xconnect
# actions (red) for one TPUv4 pod. Values were digitized from the published
# figure, so they are approximate.
#
# Job size per day is ESTIMATED from the two lines:
#   forming a torus of k cubes needs about 48 * k xconnects (Sec 3.4.1:
#   each of the 48 OCSes connects one port pair per cube), so
#   average cubes per job  ~=  xconnects / (48 * jobs)
# ---------------------------------------------------------------------------
FIG13_JOBS = [
    103, 102, 138, 169, 106, 128, 119, 98, 117, 98,
    95, 224, 181, 170, 108, 181, 248, 169, 154, 197,
    89, 94, 127, 140, 127, 186, 165, 181, 179, 120,
    153, 153, 206, 172, 107, 157, 144, 224, 140, 99,
    93, 70, 73, 115, 160, 164, 181, 166, 84, 58,
    109, 197, 113, 129, 107, 59, 84, 135, 192, 169,
]
FIG13_XCONNECTS = [
    18000, 17800, 21200, 36000, 16500, 18000, 17000, 15300, 25900, 23000,
    23000, 65400, 70400, 62500, 48600, 59500, 45400, 26900, 18000, 23200,
    10400, 17800, 20000, 21000, 14600, 22200, 25200, 24400, 22700, 17500,
    16500, 19300, 34300, 29100, 23200, 20700, 23500, 22200, 13100, 13300,
    16300, 9400, 11900, 17300, 29100, 25700, 22000, 25700, 12100, 9400,
    13600, 33300, 24000, 13600, 16800, 12300, 16000, 18800, 30600, 31400,
]

AVG_JOB_HOURS = 1.0          # assumption: paper gives no job durations
PAPER_P_MACHINE = 0.0008     # Sec 5.2: 0.08% of machines fail per day
PAPER_P_LINK = 0.00005       # Sec 5.2: 0.005% of ICI cables fail per day
SIZE_SHAPES = {1: (1, 1, 1), 2: (1, 1, 2), 4: (1, 2, 2),
               8: (2, 2, 2), 16: (2, 2, 4), 32: (2, 4, 4)}
SIZES = sorted(SIZE_SHAPES)


def size_probabilities(mean_size):
    """Probabilities over SIZES (1..32 cubes) that favour small jobs and
    have the requested average. p(size_i) ~ r**i, r found by bisection."""
    lo, hi = 1e-3, 1e3
    for _ in range(60):
        r = (lo * hi) ** 0.5
        w = np.array([r ** i for i in range(len(SIZES))])
        p = w / w.sum()
        if (p * SIZES).sum() < mean_size:
            lo = r
        else:
            hi = r
    return p


def place_background(free, size, policy, rng):
    """Try to start one background job. Returns True if it got cubes."""
    if policy == "static":
        spots = all_placements(free, SIZE_SHAPES[size])
        if not spots:
            return False
        x, y, z, a, b, c = spots[rng.integers(len(spots))]
        free[x:x+a, y:y+b, z:z+c] = False
        return True
    idx = np.argwhere(free)
    if len(idx) < size:
        return False
    for p in idx[rng.choice(len(idx), size, replace=False)]:
        free[tuple(p)] = False
    return True


PAPER_P_OCS = 0.0004         # Sec 5.2: 0.04% of OCSes fail per day


def experiment_7(trials=300, test_shape=(3, 4, 4)):
    banner("EXPERIMENT 7 - Replaying 60 days of job arrivals (Figure 13), "
           "with and without fault-tolerant routing",
           "Busy days will leave too few free cubes for a 48-cube job, even "
           "with reconfiguration. The static pod will make more normal jobs "
           "wait. Without fault-tolerant (FT) routing, a broken OCS will block "
           "every multi-cube job; FT routing will let them run slightly slower.")

    p_cube = 1 - (1 - PAPER_P_MACHINE) ** MACHINES_PER_CUBE * \
        (1 - PAPER_P_LINK) ** LINKS_PER_CUBE
    p_ocs = min(1.0, PAPER_P_OCS * REPAIR_DAYS)
    n_test = int(np.prod(test_shape))
    print(f"  cube failure prob {p_cube:.4f}, OCS failure prob {p_ocs:.4f} "
          f"(paper Sec 5.2), avg job duration assumed {AVG_JOB_HOURS} h")
    print(f"  test job = {test_shape} = {n_test} cubes")
    print(f"  policies: static | reconfig without FT routing | reconfig + FT routing\n")
    print(f"  {'day':>3} {'jobs':>5} {'avgsize':>7} {'busy':>5}   "
          f"{'wait_st':>7} {'wait_noFT':>9} {'wait_FT':>7}   "
          f"{'48_st':>6} {'48_noFT':>7} {'48_FT':>6} {'degr':>5}")

    names = ["static", "noFT", "FT"]
    rng = np.random.default_rng(SEED)
    rows = []
    for day in range(60):
        jobs, xconn = FIG13_JOBS[day], FIG13_XCONNECTS[day]
        mean_size = xconn / (NUM_OCS * jobs)
        probs = size_probabilities(mean_size)
        lam = jobs * AVG_JOB_HOURS / 24      # avg jobs running at once

        ok48 = {p: 0 for p in names}
        waited = {p: 0 for p in names}
        arrived = busy_total = degraded = 0
        for _ in range(trials):
            healthy = rng.random((4, 4, 4)) >= p_cube
            ocs_down = rng.binomial(NUM_OCS, p_ocs)
            k = rng.poisson(lam)
            sizes = rng.choice(SIZES, size=k, p=probs)
            arrived += k

            # --- static: fixed mesh, no OCS, must use contiguous blocks ---
            free = healthy.copy()
            waited["static"] += sum(not place_background(free, int(s), "static", rng)
                                    for s in sizes)
            ok48["static"] += job_can_run(free, test_shape, "static")

            # --- reconfigurable: place the jobs once (any free cubes) ---
            free = healthy.copy()
            rc_wait = sum(not place_background(free, int(s), "reconfigurable", rng)
                          for s in sizes)
            rc_fits = job_can_run(free, test_shape, "reconfigurable")
            busy_total += healthy.sum() - free.sum()

            # without FT routing: any broken OCS -> no job can be wired up
            if ocs_down == 0:
                waited["noFT"] += rc_wait
                ok48["noFT"] += rc_fits
            else:
                waited["noFT"] += k

            # with FT routing: one broken OCS is routed around (runs slower)
            if ocs_down <= 1:
                waited["FT"] += rc_wait
                ok48["FT"] += rc_fits
                degraded += ocs_down == 1
            else:
                waited["FT"] += k

        busy = busy_total / trials / 64
        w = {p: waited[p] / max(1, arrived) for p in names}
        a = {p: ok48[p] / trials for p in names}
        dg = degraded / trials
        rows.append([day, jobs, xconn, round(mean_size, 2), round(busy, 3),
                     w["static"], w["noFT"], w["FT"],
                     a["static"], a["noFT"], a["FT"], dg])
        print(f"  {day:3d} {jobs:5d} {mean_size:7.2f} {busy:5.2f}   "
              f"{w['static']:7.3f} {w['noFT']:9.3f} {w['FT']:7.3f}   "
              f"{a['static']:6.3f} {a['noFT']:7.3f} {a['FT']:6.3f} {dg:5.3f}")

    save_csv("exp7", ["day", "jobs", "xconnects", "avg_cubes_per_job",
                      "pod_busy_frac", "wait_static", "wait_reconfig_noFT",
                      "wait_reconfig_FT", "avail48_static", "avail48_reconfig_noFT",
                      "avail48_reconfig_FT", "degraded_FT"], rows)

    print("\n  SUMMARY over 60 days")
    for i, name in ((8, "48-cube job, static"), (9, "48-cube job, no FT"),
                    (10, "48-cube job, FT")):
        vals = [r[i] for r in rows]
        print(f"  {name:22s} mean={np.mean(vals):.3f}  best day={max(vals):.3f}  "
              f"worst day={min(vals):.3f}")
    for i, name in ((5, "jobs waiting, static"), (6, "jobs waiting, no FT"),
                    (7, "jobs waiting, FT")):
        print(f"  {name:22s} mean={np.mean([r[i] for r in rows]):.3f}")
    print(f"  FT routing ran degraded (1 OCS down) in "
          f"{np.mean([r[11] for r in rows])*100:.1f}% of snapshots "
          f"(paper Table 3: 0.5-8.6% slower when this happens)")

    days = [r[0] for r in rows]
    styles = [("tab:red", "-", "Static"),
              ("tab:blue", "-", "Reconfigurable, no FT routing"),
              ("tab:orange", "--", "Reconfigurable + FT routing")]
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    a1.bar(days, [r[1] for r in rows], color="tab:blue", alpha=0.5, label="Jobs/day")
    a1.set_ylabel("Jobs per day")
    a1b = a1.twinx()
    a1b.plot(days, [r[3] for r in rows], "o-", ms=3, color="tab:purple",
             label="Avg cubes/job")
    a1b.set_ylabel("Avg cubes per job")
    a1.set_title("Exp 7: Replaying Figure 13 job arrivals")
    a1.legend(loc="upper left", fontsize=8)
    a1b.legend(loc="upper right", fontsize=8)

    for (color, ls, label), i in zip(styles, (5, 6, 7)):
        a2.plot(days, [r[i] * 100 for r in rows], color=color, ls=ls, label=label)
    a2.plot(days, [r[4] * 100 for r in rows], color="gray", ls=":", label="Pod busy (%)")
    a2.set_ylabel("Jobs that must wait (%)")
    a2.grid(alpha=0.3)
    a2.legend(fontsize=8)

    for (color, ls, label), i in zip(styles, (8, 9, 10)):
        a3.plot(days, [r[i] * 100 for r in rows], color=color, ls=ls, label=label)
    a3.set_ylabel(f"{n_test}-cube job fits (%)")
    a3.set_xlabel("Day")
    a3.set_ylim(-3, 103)
    a3.grid(alpha=0.3)
    a3.legend(fontsize=8)
    save_plot(fig, "exp7")


# ---------------------------------------------------------------------------
# Experiment 7b: Experiment 7 + PREEMPTION.
#
# Same daily workloads as Experiment 7 (Figure 13). If the 48-cube job does
# not fit, the scheduler may evict (preempt) running background jobs to make
# room, as Borg does (paper Sec 3.3). Evicted jobs are assumed to restart
# later; their lost work is the COST of preemption, measured as the number of
# jobs evicted. Preemption can free cubes that are USED, but it cannot repair
# FAILED cubes.
# ---------------------------------------------------------------------------
def place_with_owner(owner, healthy, size, policy, rng, job_id):
    """Like place_background, but records which job owns each cube."""
    free = (owner == -1) & healthy
    if policy == "static":
        spots = all_placements(free, SIZE_SHAPES[size])
        if not spots:
            return False
        x, y, z, a, b, c = spots[rng.integers(len(spots))]
        owner[x:x+a, y:y+b, z:z+c] = job_id
        return True
    idx = np.argwhere(free)
    if len(idx) < size:
        return False
    for p in idx[rng.choice(len(idx), size, replace=False)]:
        owner[tuple(p)] = job_id
    return True


def preempt_reconfigurable(owner, healthy, n):
    """Evict whole jobs (largest first) until n healthy cubes are free.
    Returns number of jobs evicted, or None if impossible (too many failures)."""
    if healthy.sum() < n:
        return None
    need = n - int(((owner == -1) & healthy).sum())
    if need <= 0:
        return 0
    ids, counts = np.unique(owner[owner >= 0], return_counts=True)
    evicted = freed = 0
    for c in sorted(counts, reverse=True):
        freed += c
        evicted += 1
        if freed >= need:
            return evicted
    return None


def preempt_static(owner, healthy, shape):
    """Find the contiguous block with no FAILED cubes that needs the fewest
    evictions. Returns number of jobs evicted, or None if every block
    contains a failed cube."""
    X, Y, Z = owner.shape
    best = None
    for a, b, c in set(itertools.permutations(shape)):
        if a > X or b > Y or c > Z:
            continue
        for x in range(X - a + 1):
            for y in range(Y - b + 1):
                for z in range(Z - c + 1):
                    if not healthy[x:x+a, y:y+b, z:z+c].all():
                        continue
                    block = owner[x:x+a, y:y+b, z:z+c]
                    k = len(np.unique(block[block >= 0]))
                    if best is None or k < best:
                        best = k
                        if best == 0:
                            return 0
    return best


def experiment_7b(trials=300, test_shape=(3, 4, 4)):
    banner("EXPERIMENT 7b - Experiment 7 with preemption (Borg-style, Sec 3.3)",
           "Preemption will let the reconfigurable pod fit the 48-cube job on "
           "almost every day, because it only needs to evict enough jobs to "
           "free 48 healthy cubes. The static pod will improve less, because "
           "preemption cannot fix failed cubes inside the block, and it will "
           "need to evict more jobs to clear one contiguous block.")

    p_cube = 1 - (1 - PAPER_P_MACHINE) ** MACHINES_PER_CUBE * \
        (1 - PAPER_P_LINK) ** LINKS_PER_CUBE
    n_test = int(np.prod(test_shape))
    print(f"  cube failure prob {p_cube:.4f}, avg job duration {AVG_JOB_HOURS} h, "
          f"test job = {n_test} cubes")
    print(f"  eviction rule: evict whole jobs; reconfig evicts largest jobs first\n")
    print(f"  {'day':>3} {'jobs':>5} {'busy':>5}   {'st':>6} {'st+pre':>7} "
          f"{'rc':>6} {'rc+pre':>7}   {'evict_st':>8} {'evict_rc':>8}")

    rng = np.random.default_rng(SEED)
    rows = []
    for day in range(60):
        jobs, xconn = FIG13_JOBS[day], FIG13_XCONNECTS[day]
        probs = size_probabilities(xconn / (NUM_OCS * jobs))
        lam = jobs * AVG_JOB_HOURS / 24

        fit = {"st": 0, "st_pre": 0, "rc": 0, "rc_pre": 0}
        ev = {"st": [], "rc": []}      # jobs evicted, when preemption was used
        busy_total = 0
        for _ in range(trials):
            healthy = rng.random((4, 4, 4)) >= p_cube
            k = rng.poisson(lam)
            sizes = rng.choice(SIZES, size=k, p=probs)

            for policy, key in (("static", "st"), ("reconfigurable", "rc")):
                owner = np.full((4, 4, 4), -1)
                for j, s in enumerate(sizes):
                    place_with_owner(owner, healthy, int(s), policy, rng, j)
                free = (owner == -1) & healthy
                if key == "rc":
                    busy_total += (owner >= 0).sum()

                # without preemption
                if job_can_run(free, test_shape, policy):
                    fit[key] += 1
                    fit[key + "_pre"] += 1          # fits anyway, no eviction
                    continue
                # with preemption
                if key == "st":
                    e = preempt_static(owner, healthy, test_shape)
                else:
                    e = preempt_reconfigurable(owner, healthy, n_test)
                if e is not None:
                    fit[key + "_pre"] += 1
                    ev[key].append(e)

        busy = busy_total / trials / 64
        f = {kk: v / trials for kk, v in fit.items()}
        e_st = float(np.mean(ev["st"])) if ev["st"] else 0.0
        e_rc = float(np.mean(ev["rc"])) if ev["rc"] else 0.0
        rows.append([day, jobs, round(busy, 3), f["st"], f["st_pre"],
                     f["rc"], f["rc_pre"], round(e_st, 2), round(e_rc, 2)])
        print(f"  {day:3d} {jobs:5d} {busy:5.2f}   {f['st']:6.3f} {f['st_pre']:7.3f} "
              f"{f['rc']:6.3f} {f['rc_pre']:7.3f}   {e_st:8.2f} {e_rc:8.2f}")

    save_csv("exp7b", ["day", "jobs", "pod_busy_frac", "fit_static",
                       "fit_static_preempt", "fit_reconfig", "fit_reconfig_preempt",
                       "jobs_evicted_static", "jobs_evicted_reconfig"], rows)

    print("\n  SUMMARY over 60 days (48-cube job fits)")
    for i, label in ((3, "static"), (4, "static + preemption"),
                     (5, "reconfig"), (6, "reconfig + preemption")):
        vals = [r[i] for r in rows]
        print(f"  {label:22s} mean={np.mean(vals):.3f}  worst day={min(vals):.3f}")
    print(f"  avg jobs evicted when preempting: static="
          f"{np.mean([r[7] for r in rows if r[7] > 0]):.2f}  reconfig="
          f"{np.mean([r[8] for r in rows if r[8] > 0]):.2f}")

    days = [r[0] for r in rows]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(days, [r[3] * 100 for r in rows], color="tab:red", label="Static")
    a1.plot(days, [r[4] * 100 for r in rows], color="tab:red", ls="--",
            label="Static + preemption")
    a1.plot(days, [r[5] * 100 for r in rows], color="tab:blue", label="Reconfigurable")
    a1.plot(days, [r[6] * 100 for r in rows], color="tab:blue", ls="--",
            label="Reconfigurable + preemption")
    a1.set_ylabel(f"{n_test}-cube job fits (%)")
    a1.set_title("Exp 7b: Effect of preemption on a 48-cube job (Figure 13 workloads)")
    a1.set_ylim(-3, 103)
    a1.grid(alpha=0.3)
    a1.legend(fontsize=8, loc="center right")

    a2.plot(days, [r[7] for r in rows], color="tab:red", label="Static")
    a2.plot(days, [r[8] for r in rows], color="tab:blue", label="Reconfigurable")
    a2.set_ylabel("Jobs evicted\n(when preempting)")
    a2.set_xlabel("Day")
    a2.grid(alpha=0.3)
    a2.legend(fontsize=8)
    save_plot(fig, "exp7b")


# ---------------------------------------------------------------------------
EXPERIMENTS = {"1": experiment_1, "2": experiment_2,
               "3": experiment_3, "4": experiment_4,
               "5": experiment_5, "6": experiment_6,
               "7": experiment_7,
               # 6b: same as Exp 6 but with the full-pod 64-cube job
               "6b": lambda: experiment_6(job_shape=(4, 4, 4), name="exp6_64"),
               "7b": experiment_7b}

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    chosen = sys.argv[1:] or list(EXPERIMENTS)
    for key in chosen:
        start = time.time()
        EXPERIMENTS[key]()
        print(f"  (took {time.time() - start:.1f}s)")