"""
The smallest instance of the gap in the radial reduction (Lemma "Monotonicity
in the contact radii" and its use in Proposition "Polar form" and Corollary
"The multi-direction conjecture"): two neighbours at distance 2.4, inside the
radius 2R = 2 sqrt(3/2) = 2.449... at which the truncated estimates of the
paper cut the cell, and as close in angle as packing validity allows.
"""
import math
import numpy as np

d = 2.4
c = (2 * d * d - 4) / (2 * d * d)            # least admissible angle: |y1 - y2| = 2
th = math.acos(c)
y1 = d * np.array([1.0, 0, 0, 0])
y2 = d * np.array([math.cos(th), math.sin(th), 0, 0])
print("neighbours at distance %.3f, %.3f; 2R = 2 sqrt(3/2) = %.6f" % (np.linalg.norm(y1), np.linalg.norm(y2), 2 * math.sqrt(1.5)))
print("distance between them: %.9f  (a valid packing: >= 2)" % np.linalg.norm(y1 - y2))
print("angle between their directions: %.4f degrees; inner product %.6f" % (math.degrees(th), c))
print("at the all-contact corner both sit at distance 2, %.6f apart: not a packing," % (2 * math.sqrt(2 - 2 * c)))
print("and the directions are not a contact configuration (inner product %.4f > 1/2)" % c)
