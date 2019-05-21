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
            # TODO cmult
            cgp_inv,
            cgp_abs,
            cgp_sqrt,
            # TODO cgp_cpow
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
            # TODO split_before
            # TODO split_after
            # TODO range_in
            cgp_index_y,
            # TODO index_p
            cgp_vectorize,
            cgp_first,
            cgp_last,
            cgp_diff,
            cgp_avg_diff,
            # TODO rotate
            cgp_reverse,
            cgp_push_back,
            cgp_push_front,
            cgp_set,
            cgp_sum,
            cgp_transpose,
            cgp_vecfromdouble,
############################## Miscellaneous functions ##############################
            cgp_nop,
            # TODO const
            # TODO cgp_constvectord = Primitive("constvectord", lambda x: , 1)
            cgp_zeros,
            cgp_ones,

            # Three input variables: R, G and B matrices.
            Symbol("R"),
            Symbol("G"),
            Symbol("B"),

            # TODO define this.
            # Ephemeral constant is a function returning random value number.
            #Ephemeral("random", rng.normal)
            # Symbolic constant represents constant in the graph, we should have one for every function that needs one.
            #Constant("C1"),
            #Constant("C2"),
            #Constant("C3"),
            #Constant("C4")

            # ??? Structural constants ??? We probably don't need this.
            #Structural("structural", lambda x, y: max(x, y))
        ]
primitiveSet = PrimitiveSet.create(primitives)