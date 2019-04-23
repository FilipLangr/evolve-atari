import numpy as np
from cartesian.cgp import Primitive

#################################################################
# Here we define pool of functions available for CGP experiments.
#################################################################

def cgp_inner_set(x, y):
    if isinstance(x, float):
        # x is scalar
        if isinstance(y, float):
            # both x and y are scalars
            return x
        else:
            # x is scalar, y is array
            return np.array([x] * y.reshape(-1).shape[0])
    else:
        # x is array
        if isinstance(y, float):
            # x is array, y is scalar
            return np.array([y] * x.reshape(-1).shape[0])
        else:
            # both x and y are arrays, use mean of x.
            return np.array([np.mean(x)] * y.reshape(-1).shape[0])


# Helper functions.
def float2index(matrix, index):
    vector = np.array(matrix).reshape(-1)
    l = vector.shape[0]
    index = np.abs(np.mean(index))
    index *= l
    index = np.max(np.min(l - 1, index), 0)
    return index

# TODO review.
cgp_add = Primitive("add", lambda x, y: (x + y)/2, 2)
cgp_aminus = Primitive("subtract", lambda x, y: np.abs(x - y)/2, 2)
cgp_mult = Primitive("multiply", lambda x, y: x * y, 2)
cgp_inv = Primitive("invert", lambda x: 1 / (x+0.00001), 1)

# List processing functions.
# TODO ReDO to isinstace(x, np.ndarray)
# TODO parameter ones.
cgp_index_y = Primitive("index_y", lambda x, y: x if isinstance(x, float) else np.array(x).reshape(-1)[float2index(x, (y+1) / 2)], 2)
cgp_first = Primitive("first", lambda x: np.array(x).reshape(-1)[0], 1)
cgp_last = Primitive("last", lambda x: np.array(x).reshape(-1)[-1], 1)
# TODO differences and avg_differences.
cgp_reverse = Primitive("reverse", lambda x: x if isinstance(x, float) else np.array(x).reshape(-1)[::-1], 1)
cgp_push_back1 = Primitive("push_back1", lambda x, y: np.append(np.array(x).reshape(-1), np.array(y).reshape(-1)), 2)
cgp_push_back1 = Primitive("push_back1", lambda x, y: np.append(np.array(y).reshape(-1), np.array(x).reshape(-1)), 2)
cgp_set = Primitive("set", cgp_inner_set, 2)
cgp_sum = Primitive("sum", lambda x: np.sum(x), 1)
cgp_transpose = Primitive("transpose", lambda x: x if isinstance(x, float) else np.transpose(x), 1)
cgp_vecfromdouble= Primitive("vecfromdouble", lambda x: np.array(x) if isinstance(x, float) else x, 1)