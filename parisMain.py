import numpy as np
import math
import parisFunctions as pf
import acquisition_function as acq
import create_random_matrix as crm
import chooseBoundedRegion as cbr
import scipy.spatial.distance as sp


D=64
d=3

# Step 1
# Generate random matrix
A = crm.random_matrix(D, d)


# Step 2
# Choose bounded region set
regionBound = math.sqrt(d)
regionBoundStepSize = 0.1

#Y = N x d matrix
Y = cbr.chooseBoundedRegion(d, -regionBound, regionBound, regionBoundStepSize)

covarianceMatrix = np.ones([10,10])
#t = number of iterations. 
numIter=10


#k=acq.sqexp_kernel([1,2,3],[1,2,3])

for t in range(0, numIter):
	covarianceMatrix[t,:] = t
	covarianceMatrix[:,t] =t
print covarianceMatrix
 

