from atari_primitive_set import primitives
from cartesian.cgp import Symbol
import numpy as np

# Tests CGP functions in atari_primitive_set for various inputs.

scalars = [-1.31, 0, 0.13, 1, 1.32, 3324]
arrays = [
    np.random.rand(3, 4) * 200 - 100,
    np.random.rand(400, 2) * 200 - 100,
    np.random.rand(12,1) * 200 - 100,
    np.random.rand(3, ) * 200 - 100,
    np.random.rand(1, 12) * 200 - 100,
    np.random.rand(1, 1) * 200 - 100,
    np.random.rand(1, ) * 200 - 100,
    np.random.rand(5, 5) * 200 - 100,
    ]
all = scalars + arrays

def test_fce(primitive):
    f = primitive.function
    if primitive.arity == 2:
        for i in range(len(all)):
            for j in range(i+1, len(all)):
                valA = all[i]
                valB = all[j]
                f(valA, valB)
                f(valB, valA)
                f(valA, valA)
                f(valB, valB)

    elif primitive.arity == 1:
        for item in all:
            f(item)
    else:
        raise RuntimeError("Should not happen")
    print("%s passed the test." % primitive.name)

if __name__ == "__main__":
    for p in primitives:
        if not isinstance(p, Symbol):
            test_fce(p)
    print("ALL FUNCTIONS PASSED THE TEST!")