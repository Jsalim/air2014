# utility imports
import numpy as np
import math
from get_similar_point import get_similar_point

# REMBO imports
import create_random_matrix as crm
import choose_bounded_region as cbr

# LEROT/L2R imports
import Learning2Rank

# REMBO init
D=10
d=3
region_bound = math.sqrt(d)
region_bound_stepsize = 0.5

# DBGD init
probe_magnitude = 0.5 # just arbitrary number for now..
change_magnitude = 0.01

# L2R init (create Learning2Rank class instance)
l2r = Learning2Rank('../../data/NP2004/Fold1/test.txt', D)
	
# ###### REMBO + LEROT ####### #

# Step 1
# Generate random matrix
A = crm.random_matrix(D, d)

# Step 2
# Choose bounded region set: y = N x d matrix (the exhaustive search subset)
y = cbr.choose_bounded_region(d, -region_bound, region_bound, region_bound_stepsize)
y = np.array(y)

# start with random sample
Y = acq.select_random_point(y)

# t starts from 1 to avoid having t+1 all over the code (might confuse us.)
# is our time the length of the queries we do? We could shorten it for testing at least.
for t in range(1, l2r.get_query_length):
	# create y_probe based on y_current
	y_probe, altered_dimension = get_similar_point(y_current, probe_magnitude, region_bound)
	
	# prep for L2R: convert to D-dimensional
	Ay_current, Ay_probe =  l2r.prepare_rankers(A, y_current, y_probe)
	
	# evaluate
	result = l2r.evaluate(Ay_current, Ay_probe, q)
	
	# if Ay_probe better than Ay_current, add new point in Y based on Ay_probe
	if result > 0:
		y_probe_fraction = y_probe # we only move towards y_probe a fraction of where we checked. Do we need this? The paper does it...
		y_probe_fraction[altered_dimension] = y_probe[altered_dimension] + (y_probe[altered_dimension]-y_current[altered_dimension])*change_magnitude
		Y = np.append(Y,y_probe_fraction,0)
	
	# I dont think we need to delete stuff from y because we actually only use it the very first time... still looking into this.