from cartesian.cgp import *
from cgp_functions import *

##################################################################
# Here we define a specific set of functions for a CGP experiment.
##################################################################

primitives = [
            # TODO ...
            # A set of functions (defined in cgp_functions.py) goes here.
            #cgp_add,
            #cgp_aminus,
            #cgp_mult,
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