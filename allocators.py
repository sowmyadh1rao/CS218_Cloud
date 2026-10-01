import itertools
import numpy as np

def static_allocate(free, shape):
    """TPUv3-style: need a CONTIGUOUS all-healthy box of this shape.
    Rotations allowed, no wrap-around. Returns list of positions or None."""
    X, Y, Z = free.shape
    for a, b, c in set(itertools.permutations(shape)):
        if a > X or b > Y or c > Z:
            continue
        for x in range(X - a + 1):
            for y in range(Y - b + 1):
                for z in range(Z - c + 1):
                    if free[x:x+a, y:y+b, z:z+c].all():
                        return [(x+i, y+j, z+k) for i in range(a)
                                for j in range(b) for k in range(c)]
    return None

def reconfigurable_allocate(free, n):
    """TPUv4-style: OCS can wire ANY n healthy cubes together."""
    idx = np.argwhere(free)
    return [tuple(p) for p in idx[:n]] if len(idx) >= n else None