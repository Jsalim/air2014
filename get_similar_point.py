from random import randint
from random import random

def get_similar_point(old_point, magnitude, region_bound):
	new_point = old_point
	probe_dimension = randint(d)
	altered_value = new_point[probe_dimension]
	if (random() < 0.5):
		altered_value = altered_value + magnitude
		altered_value = min(altered_value, region_bound)
	else:
		altered_value = altered_value - magnitude
		altered_value = max(altered_value, -region_bound)
	new_point[probe_dimension] = altered_value
	return new_point, probe_dimension
	