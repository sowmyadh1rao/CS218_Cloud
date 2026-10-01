from pod import State
from allocators import static_allocate, reconfigurable_allocate


class Borg:
    """Simple scheduler: gives a job any free healthy cubes (TPUv4-style),
    otherwise puts it in a waiting queue."""
    def __init__(self, pod, policy="reconfigurable"):
        self.pod = pod
        self.policy = policy
        self.queue = []      # jobs waiting for cubes
        self.running = {}    # job_id -> Job
        self.policy=policy

    def submit(self, job):
        """A new job arrives at Borg."""
        print(f"[Borg] {job.job_id} arrives, needs {job.num_cubes} cubes")
        if not self.try_schedule(job):
            self.queue.append(job)
            print(f"[Borg] {job.job_id} QUEUED (not enough free cubes)")

    def try_schedule(self, job):
        free = self.pod.free_mask()  # 4x4x4 True/False grid

        if self.policy == "static":
            positions = static_allocate(free, job.shape)
        else:
            positions = reconfigurable_allocate(free, job.num_cubes)

        if positions is None:
            return False

        chosen = [self.pod.grid[p] for p in positions]
        for cube in chosen:
            cube.state = State.USED
        job.cubes = chosen
        job.status = "RUNNING"
        self.running[job.job_id] = job
        print(f"[Borg] {job.job_id} RUNNING on {len(chosen)} cubes ({self.policy})")
        return True

    def finish(self, job_id):
        """A job completes: free its cubes, then retry waiting jobs."""
        job = self.running.pop(job_id)
        for cube in job.cubes:
            if cube.state == State.USED:
                cube.state = State.HEALTHY
        job.status = "DONE"
        print(f"[Borg] {job_id} DONE, released {len(job.cubes)} cubes")
        self._retry_queue()

    def _retry_queue(self):
        still_waiting = []
        for job in self.queue:
            if not self.try_schedule(job):
                still_waiting.append(job)
        self.queue = still_waiting
