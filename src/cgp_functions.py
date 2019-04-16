import numpy as np
from cartesian.cgp import Primitive

#################################################################
# Here we define pool of functions available for CGP experiments.
#################################################################

# TODO ...
cgp_add = Primitive("add", lambda x, y: (x + y)/2, 2)
cgp_aminus = Primitive("subtract", lambda x, y: np.abs(x - y)/2, 2)
cgp_mult = Primitive("multiply", lambda x, y: x * y, 2)
cgp_inv = Primitive("invert", lambda x: 1 / (x+0.00001), 1)