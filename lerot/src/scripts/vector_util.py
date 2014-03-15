from random import randint
from random import random
import numpy as np

# takes a point in a random direction and returns this. This will become y_probe
def get_random_close_point(old_point, magnitude, min_region_bound, max_region_bound, d):
	new_point = np.copy(old_point)
	while np.all(np.equal(new_point, old_point)):		
		# get random dimension which we change
		probe_dimension = randint(0,d-1)
		altered_value = new_point[0,probe_dimension]
		
		# for any cases where the value we alter is already (nearly) equal to the edge
		# changes the value away from the edge
		if float_equals(altered_value, min_region_bound):
			altered_value = altered_value + magnitude
			altered_value = min(altered_value, max_region_bound)
		elif float_equals(altered_value, max_region_bound):
			altered_value = altered_value - magnitude
			altered_value = max(altered_value, min_region_bound)
			
		# for any non-edge cases. Most cases will use this part
		# changes the value to a random side
		elif (random() < 0.5):
			altered_value = altered_value + magnitude
			altered_value = min(altered_value, max_region_bound)
		else:
			altered_value = altered_value - magnitude
			altered_value = max(altered_value, min_region_bound)
		new_point[0,probe_dimension] = altered_value
	return new_point, probe_dimension

# test for float equality
def float_equals(v1,v2):
	return abs(v1-v2) < 0.001
	
# generates a random point out of the grid
def get_random_point(y):
	result = []
	idx = randint(0, len(y)-1)
	result.append(y[idx])
	return np.matrix(result)
	
# move a tiny fraction to the probed point, which you decided was better than the current point
def move_towards_new_point(y_old, y_probed, dimension, magnitude):
	y_new = np.copy(y_old)
	y_new[0,dimension] = y_old[0,dimension] - (y_old[0,dimension]-y_probed[0,dimension])*magnitude
	return y_new