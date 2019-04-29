from cartesian.cgp import *
from cgp_functions import *

##################################################################
# Here we define a specific set of functions for a CGP experiment.
##################################################################

primitives = [
            classic_add,
            classic_mult,

            Symbol("inp01"),
            Symbol("inp02"),
            Symbol("inp03"),
            Symbol("inp04"),
            #Constant("C1"),
        ]
primitiveSet = PrimitiveSet.create(primitives)