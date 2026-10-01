# TPUv4 Availability Simulator

A small Python simulator for the resource-allocation and availability problem in
*Resiliency at Scale: Managing Google's TPUv4 Machine Learning Supercomputer*
(Zu et al., NSDI '24). It compares a **static** (TPUv3-style, contiguous) and a
**reconfigurable** (TPUv4-style, OCS) allocation policy on a 4 x 4 x 4 pod of 64 cubes.

No cloud resources, GPUs, or ML frameworks are needed; it runs on a normal laptop.

## Run on Ubuntu

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git
git clone <your-repo-url>
cd <repo-folder>
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run all experiments (about 2 minutes):

```bash
python3 experiments.py
```

Run selected experiments only:

```bash
python3 experiments.py 1 3        # experiments 1 and 3
python3 experiments.py 6b 7b      # variants
```

Graphs (`expN.png`) and tables (`expN.csv`) are saved in the `results/` folder.
A fixed random seed makes the results reproducible.

Quick scheduler demo (one job, both policies):

```bash
python3 test_borg.py
```

## Experiments

| ID | What it tests |
|----|---------------|
| 1  | Job size (1-64 cubes), empty and 20%-occupied pod |
| 2  | Per-machine availability (99%-99.99%) |
| 3  | Fragmentation: packed vs scattered background jobs |
| 4  | Pod grows, job is always half the pod |
| 5  | Pod grows, job fixed at 32 cubes (spares) |
| 6  | 100 weeks of failure rates from the paper's Figure 12, 48-cube job |
| 6b | Same as 6 with a 64-cube (full-pod) job |
| 7  | 60 days of workloads from the paper's Figure 13, with/without fault-tolerant routing |
| 7b | Experiment 7 with preemption |

## Files

| File | Purpose |
|------|---------|
| `pod.py` | Pod of cubes; each cube is HEALTHY, FAILED, or USED |
| `allocators.py` | Static and reconfigurable allocation policies |
| `job.py` | A job requesting a box-shaped set of cubes |
| `borg.py` | Simple scheduler with a waiting queue |
| `montecarlo.py` | Basic availability estimate (random trials) |
| `experiments.py` | All experiments; writes results to `results/` |
| `test_borg.py` | Small scheduler demo |

## Notes

Values from the paper's Figures 12 and 13 were read from the published graphs and are approximate.
AI assistance was used to generate parts of the code; the simulator design and experiments are my own.