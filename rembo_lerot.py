import numpy as np
import math
import create_random_matrix as crm
import choose_bounded_region as cbr
from get_similar_point import get_similar_point
import environment, comparison, query

# REMBO init
D=10
d=3

# DBGD init
probe_magnitude = 0.5 # just arbitrary number for now..
change_magnitude = 0.01
# L2R init
test_num_features = 64
queries = query.load_queries('../../data/NP2004/Fold1/test.txt', 64)
bi = comparison.BalancedInterleave()
user_model = environment.CascadeUserModel('--p_click 0:0.0,1:1 --p_stop 0:0.0,1:0.0')	
	
# ###### REMBO + LEROT ####### #

# Step 1
# Generate random matrix
A = crm.random_matrix(D, d)

# Step 2
# Choose bounded region set
region_bound = math.sqrt(d)
region_bound_stepsize = 0.5

#y = N x d matrix (the exhaustive search subset)
y = cbr.choose_bounded_region(d, -region_bound, region_bound, region_bound_stepsize)
y = np.array(y)

max_iter = 10

# start with random sample
Y = acq.select_random_point(y)

# t starts from 1 to avoid having t+1 all over the code (might confuse us.)
# is our time the length of the queries we do? We could shorten it for testing at least.
for t in range(1, len(queries)):
	# create y_probe based on y_current
	y_probe, altered_dimension = get_similar_point(y_current, probe_magnitude, region_bound)
	
	# prep for L2R: convert to D-dimensional
	Ay_current = np.dot(A,(y_current.T)).T
	Ay_probe = np.dot(A,(y_probe.T)).T
	
	# evaluate (to do: make into multiple functions)
	q = queries[random.choice(queries.keys())]
	list, context = bi.interleave(Ay_current, Ay_probe, q, 10)
	clicks = user_model.get_clicks(list,q.get_labels())
	result = bi.infer_outcome(list,context,clicks,q)
	if result < 0:
		continue
	else:
		y_probe_fraction = y_probe # we only move towards y_probe a fraction of where we checked. Do we need this? The paper does it...
		y_probe_fraction[altered_dimension] = y_probe[altered_dimension] + (y_probe[altered_dimension]-y_current[altered_dimension])*change_magnitude
		Y = np.append(Y,y_probe_fraction,0)
	
	# I dont think we need to delete stuff from y because we actually only use it the very first time... still looking into this.