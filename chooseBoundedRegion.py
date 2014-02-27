import numpy as np

# range() function, but for floats and not integers. 
# x = min, y = max, step = stepsize. 
# Note that y is INCLUSIVE
def floatrange(x, y, step):
	while x <= y:
		yield x
		x += step

#step2
# d = dimensions
# lowerBound/upperBound are the bounds, such as -sqrt(d) and sqrt(d)
# stepSize = the interval with which we traverse the box, such as 0.01
# example usage: chooseBoundedRegion(3, 0, 1, 0.01)
def chooseBoundedRegion (d, lowerBound, upperBound, stepSize):
	YLength = int( round( (upperBound-lowerBound+stepSize)/stepSize ) )**d
	Y = np.zeros((YLength,d))
	index = 0
	for i in floatrange(lowerBound, upperBound, stepSize):
		for j in floatrange(lowerBound, upperBound, stepSize):
			for k in floatrange(lowerBound, upperBound, stepSize):
				Y[index] = [i,j,k]
				index = index+1
	return Y