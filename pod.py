"""
Starter simulator: a TPUv4-style pod of 64 cubes arranged in a 4x4x4 grid.
Each cube is HEALTHY (0), FAILED (1), or USED (2) by another job.
"""
from enum import IntEnum
import numpy as np

class State(IntEnum):
    HEALTHY = 0   # free and working
    FAILED = 1    # broken / under maintenance
    USED = 2      # occupied by another job


class Cube:
    def __init__(self, cube_id, position, state=State.HEALTHY):
        self.cube_id = cube_id
        self.position = position          # (x, y, z) in the pod grid
        self.state = State(state)

    def is_available(self):
        return self.state == State.HEALTHY

    def __repr__(self):
        return f"Cube(id={self.cube_id}, pos={self.position}, state={self.state.name})"


class Pod:
    def __init__(self, shape=(4, 4, 4)):
        self.shape = shape
        self.grid = np.empty(shape, dtype=object)
        for cube_id, pos in enumerate(np.ndindex(shape)):
            self.grid[pos] = Cube(cube_id, pos)

    def cubes(self):
        """All cubes as a flat list."""
        return list(self.grid.flat)

    def reset_all_healthy(self):
        """Set every cube in the pod back to HEALTHY."""
        for cube in self.cubes():
            cube.state = State.HEALTHY

    def free_mask(self):
        return self.state_matrix() == State.HEALTHY

    def randomize(self, p_fail, p_used, rng):
        """Give every cube a random state.
        p_fail = chance a cube is failed, p_used = chance it is occupied."""
        for cube in self.cubes():
            r = rng.random()
            if r < p_fail:
                cube.state = State.FAILED
            elif r < p_fail + p_used:
                cube.state = State.USED
            else:
                cube.state = State.HEALTHY

    def free_mask(self):
        return self.state_matrix() == State.HEALTHY  # 4x4x4 True/False

    def state_matrix(self):
        """4x4x4 array of state numbers (0/1/2) - handy for allocation checks."""
        return np.vectorize(lambda c: int(c.state))(self.grid)

    def count_states(self):
        m = self.state_matrix()
        return {s.name: int((m == s).sum()) for s in State}

    def print_layers(self):
        """Print the pod one z-layer at a time."""
        m = self.state_matrix()
        for z in range(self.shape[2]):
            print(f"z = {z}")
            print(m[:, :, z])
            print()


if __name__ == "__main__":
    pod = Pod(shape=(4, 4, 4))
    pod.reset_all_healthy()      # make sure every cube is HEALTHY

    print("Total cubes:", len(pod.cubes()))
    print("Example cube:", pod.grid[1, 2, 3])

    print("\nState counts:", pod.count_states())
    print("\nPod layout (0=healthy, 1=failed, 2=used):\n")
    pod.print_layers()