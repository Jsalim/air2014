import numpy as np
from itertools import product
# range() function, but for floats and not integers.
# x = min, y = max, step = stepsize.
# Note that y is INCLUSIVE
def floatrange(x, y, step):
	while x <= y:
		yield x
		x += step

#step2
# d = dimensions
# lower_bound/upper_bound are the bounds, such as -sqrt(d) and sqrt(d)
# step_size = the interval with which we traverse the box, such as 0.01

# example usage: choose_bounded_region(3, 0, 1, 0.01) yields y for 3 dimensions, 
# with all possible combinations between 0 (inclusive) and 1 (inclusive) 
# with a stepsize of 0.01
def choose_bounded_region (d, lower_bound, upper_bound, step_size):
	# yLength = int( round( (upper_bound-lower_bound+step_size)/step_size ) )**d
	# print yLength
	steps_array = []
	for val in floatrange(lower_bound,upper_bound,step_size):
		steps_array.append(val)
	y = []
	for vec in product(steps_array, repeat=d):
		arr = []
		for val in vec:
			arr.append(val)
		y.append(arr)
	return y

y = choose_bounded_region(3,0,1,0.2)
print y