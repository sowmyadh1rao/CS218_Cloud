class Job:
    """A training job asking for a number of cubes."""

    def __init__(self, job_id, num_cubes,shape):
        self.job_id = job_id
        self.shape = shape
        self.num_cubes = num_cubes
        self.cubes = []  # cubes assigned to this job
        self.status = "PENDING"  # PENDING -> RUNNING -> DONE

    def __repr__(self):
        return f"Job({self.job_id}, cubes={self.num_cubes}, {self.status})"