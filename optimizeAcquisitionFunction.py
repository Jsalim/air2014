#step4 

import numpy as np


#X=box t=current timestep D=No. of features
def optimizeAcquisitionFunction(t,X,D):

	#initialize the max
	maxX = X.item(0);

	for(x in X):
		#u = aqcuisition function
		y = u(x,D)

		#check for max
		if(y>maxX):
			maxX=y

	#maxX is the x that optimizes the u function
	return maxX



