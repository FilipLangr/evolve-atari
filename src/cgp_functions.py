import numpy as np
from cartesian.cgp import Primitive
from scipy.stats import skew, kurtosis

#################################################################
# Here we define pool of functions available for CGP experiments.
#################################################################

############################## Utils functions ##############################

def float2index(vector, y):
    """
    Transform float y representation to (integer) index to vector.
    :param vector: Strictly one-dim np.array.
    :param y: Any np.array or scalar.
    :return: Index to vector.
    """
    l = vector.shape[0]
    index_f = np.mean(np.abs((y + 1) / 2))
    return np.min(np.max((l-1) * index_f, 0.0), l-1)

def common_submatrices(x, y):
    (x1, x2), (y1, y2) = x.shape, y.shape
    n1, n2 = min(x1, y1), min(x2, y2)
    return x[0:n1, 0:n2], y[0:n1, 0:n2]

def scaled_array(array):
    array[~np.isfinite(array)] = 0.0
    array[array < -1.0] = -1.0
    array[array > 1.0] = 1.0
    return array

def scaled_scalar(number):
    if not np.isfinite(number):
        return 0.0
    else:
        return np.min(np.max(number, -1.0), 1.0)

############################## Mathematical functions ##############################

def cgp_inner_ypow(x, y):
    if (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
        x, y = common_submatrices(x, y)
    return np.pow(np.abs(x), np.abs(y))

def cgp_inner_sqrtxy(x, y):
    if (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)):
        x, y = common_submatrices(x, y)
    return np.sqrt(x**2 + y**2) / np.sqrt(2)

cgp_add = Primitive("add", lambda x, y: sum(*common_submatrices(x, y)) / 2.0 if (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)) else (x+y)/2.0, 2)
cgp_aminus = Primitive("aminus", lambda x, y: np.abs(sum(*common_submatrices(x, -y))) / 2.0 if (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)) else np.abs(x-y)/2.0, 2)
cgp_mult = Primitive("mult", lambda x, y: np.mult(*common_submatrices(x, y)) if (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)) else x*y, 2)
# TODO cmult
cgp_inv = Primitive("inverse", lambda x: scaled_array(np.divide(1, x)) if isinstance(x, np.ndarray) else scaled_scalar(np.divide(1, x)), 1)
cgp_abs = Primitive("abs", lambda x: np.abs(x), 1)
cgp_sqrt = Primitive("sqrt", lambda x: np.sqrt(np.abs(x)), 1)
# TODO cgp_cpow
cgp_ypow = Primitive("ypow", cgp_inner_ypow, 2)
cgp_expx = Primitive("expx", lambda x: (np.exp(x) - 1) / (np.exp(1) - 1), 1)
cgp_sinx = Primitive("sinx", lambda x: np.sin(x), 1)
cgp_cosx = Primitive("cosx", lambda x: np.cos(x), 1)
cgp_sqrtxy = Primitive("sqrtxy", cgp_inner_sqrtxy, 2)
cgp_acos = Primitive("acos", lambda x: np.arccos(x) / np.pi, 1)
cgp_asin = Primitive("asin", lambda x: 2*np.arcsin(x) / np.pi, 1)
cgp_atan = Primitive("atan", lambda x: 4*np.arctan(x) / np.pi, 1)

############################## Statistical functions ##############################

cgp_std = Primitive("std", lambda x: scaled_scalar(np.std(x)) if isinstance(x, np.ndarray) else x, 1)
cgp_skew = Primitive("skew", lambda x: scaled_scalar(skew(x.reshape(-1))) if isinstance(x, np.ndarray) else x, 1)
cgp_kurtosis = Primitive("kurtosis", lambda x: scaled_scalar(kurtosis(x.reshape(-1))) if isinstance(x, np.ndarray) else x, 1)
cgp_mean = Primitive("mean", lambda x: np.mean(x) if isinstance(x, np.ndarray) else x, 1)
cgp_range = Primitive("range", lambda x: np.max(x) - np.min(x) - 1 if isinstance(x, np.ndarray) else x, 1)
cgp_round = Primitive("round", lambda x: np.round(x), 1)
cgp_ceil = Primitive("ceil", lambda x: np.ceil(x), 1)
cgp_floor = Primitive("floor", lambda x: np.floor(x), 1)
cgp_max1 = Primitive("max1", lambda x: np.max(x), 1)
cgp_min1 = Primitive("min1", lambda x: np.min(x), 1)

############################## Comparison functions ##############################

cgp_lt = Primitive("lower_than", lambda x, y: np.less(*common_submatrices(x, y)).astype(float) if (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)) else (x < y).astype(float), 2)
cgp_gt = Primitive("greater_than", lambda x, y: np.greater(*common_submatrices(x, y)).astype(float) if (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)) else (x > y).astype(float), 2)
cgp_max2 = Primitive("max2", lambda x, y: np.maximum(*common_submatrices(x, y)).astype(float) if (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)) else (np.maximum(x, y)), 2)
cgp_min2 = Primitive("min2", lambda x, y: np.minimum(*common_submatrices(x, y)).astype(float) if (isinstance(x, np.ndarray) and isinstance(y, np.ndarray)) else (np.minimum(x, y)), 2)

############################## List processing functions ##############################

def cgp_inner_set(x, y):

    if isinstance(y, np.ndarray):
        return np.mean(x) * np.ones(shape=y.shape)
    elif isinstance(x, np.ndarray):
        return y * np.ones(shape=x.shape)
    else:
        return x

def cgp_inner_index_y(x, y):
    if isinstance(x, np.ndarray):
        vector = x.reshape(-1)
        ind = float2index(vector, y)
        return vector[ind]
    else:
        return x

# TODO split_before
# TODO split_after
# TODO range_in
cgp_index_y = Primitive("index_y", cgp_inner_index_y, 2)
# TODO index_p
cgp_vectorize = Primitive("vectorize", lambda x: x.reshape(-1) if isinstance(x, np.ndarray) else x, 1)
cgp_first = Primitive("first", lambda x: np.array(x).reshape(-1)[0], 1)
cgp_last = Primitive("last", lambda x: np.array(x).reshape(-1)[-1], 1)
cgp_diff = Primitive("diff", lambda x: scaled_array(np.diff(x.reshape(-1))) if isinstance(x, np.ndarray) else x, 1)
cgp_avg_diff = Primitive("avg_diff", lambda x: scaled_scalar(np.mean(np.diff(x.reshape(-1)))) if isinstance(x, np.ndarray) else x, 1)
# TODO rotate
cgp_reverse = Primitive("reverse", lambda x: np.array(x).reshape(-1)[::-1] if isinstance(x, np.ndarray) else x, 1)
cgp_push_back = Primitive("push_back", lambda x, y: np.append(np.array(x).reshape(-1), np.array(y).reshape(-1)), 2)
cgp_push_front = Primitive("push_front", lambda x, y: np.append(np.array(y).reshape(-1), np.array(x).reshape(-1)), 2)
cgp_set = Primitive("set", cgp_inner_set, 2)
cgp_sum = Primitive("sum", lambda x: scaled_scalar(np.sum(x)), 1)
cgp_transpose = Primitive("transpose", lambda x: np.transpose(x) if isinstance(x, np.ndarray) else x, 1)
cgp_vecfromdouble= Primitive("vecfromdouble", lambda x: x if isinstance(x, np.ndarray) else np.array(x), 1)

############################## Miscellaneous functions ##############################

cgp_nop = Primitive("nop", lambda x: x, 1)
# TODO const
# TODO cgp_constvectord = Primitive("constvectord", lambda x: , 1)
cgp_zeros = Primitive("zeros", lambda x: np.zeros(shape=x.shape) if isinstance(x, np.ndarray) else x, 1)
cgp_ones = Primitive("ones", lambda x: np.ones(shape=x.shape) if isinstance(x, np.ndarray) else x, 1)
