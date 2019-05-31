from cartesian.cgp import *
from cgp_functions import *

##################################################################
# Here we define a specific set of functions for a CGP experiment.
##################################################################

primitives = [
            # A set of functions (defined in cgp_functions.py) goes here.
############################## Mathematical functions ##############################
            cgp_add,
            cgp_aminus,
            cgp_mult,
            cgp_inv,
            cgp_abs,
            cgp_sqrt,
            cgp_ypow,
            cgp_expx,
            cgp_sinx,
            cgp_cosx,
            cgp_sqrtxy,
            cgp_acos,
            cgp_asin,
            cgp_atan,
############################## Statistical functions ##############################
            cgp_std,
            cgp_skew,
            cgp_kurtosis,
            cgp_mean,
            cgp_range,
            cgp_round,
            cgp_ceil,
            cgp_floor,
            cgp_max1,
            cgp_min1,
############################## Comparison functions ##############################
            cgp_lt,
            cgp_gt,
            cgp_max2,
            cgp_min2,
############################## List processing functions ##############################
            cgp_index_y,
            cgp_vectorize,
            cgp_first,
            cgp_last,
            cgp_diff,
            cgp_avg_diff,
            cgp_reverse,
            cgp_push_back,
            cgp_push_front,
            cgp_set,
            cgp_sum,
            cgp_transpose,
            cgp_vecfromdouble,
############################## Miscellaneous functions ##############################
            cgp_nop,
            cgp_zeros,
            cgp_ones,

            # Three input variables: R, G and B matrices.
            Symbol("R"),
            Symbol("G"),
            Symbol("B"),
        ]
primitiveSet = PrimitiveSet.create(primitives)