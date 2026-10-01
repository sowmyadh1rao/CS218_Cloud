import numpy as np
from pod import Pod
from allocators import static_allocate, reconfigurable_allocate

def cube_fail_prob(p_machine):
    """A cube = 16 machines; it's unusable if ANY machine is down (paper §2.2)."""
    return 1 - (1 - p_machine) ** 16

def availability(job_shape, p_fail, p_used, trials=2000,
                 pod_shape=(4, 4, 4), seed=0):
    rng = np.random.default_rng(seed)
    pod = Pod(pod_shape)
    n = int(np.prod(job_shape))
    ok = {"static": 0, "reconfigurable": 0}
    for _ in range(trials):
        pod.randomize(p_fail, p_used, rng)
        free = pod.free_mask()
        ok["static"] += static_allocate(free, job_shape) is not None
        ok["reconfigurable"] += reconfigurable_allocate(free, n) is not None
    return {k: v / trials for k, v in ok.items()}