import numpy as np
import math
import acquisition_function as acq
import create_random_matrix as crm
import choose_bounded_region as cbr
import scipy.spatial.distance as sp


D=10
d=3

# Step 1
# Generate random matrix
A = crm.random_matrix(D, d)


# Step 2
# Choose bounded region set
regionBound = math.sqrt(d)
regionBoundStepSize = 0.5

#y = N x d matrix (the exhaustive search subset)
y = cbr.choose_bounded_region(d, -regionBound, regionBound, regionBoundStepSize)

#create artificial data [0...1]
#data = cbr.choose_bounded_region(d,0,1,0.5)
#y_projected = acq.projection(data,ybest)



max_iter = 10


# t starts from 1 to avoid having t+1 all over the code (might confuse us.)
for t in range(1, max_iter+1):
	  
	#number of samples = t for each iteration step. (is that correct?)
  	number_of_samples =  t


  	#Step 3 : sample out of y , and create Y
  	Y = acq.select_sample_set(number_of_samples,y)
  	y=np.array(y)


  	candidates=[]


  	#get the K inverse matrix
  	Kinv = acq.get_Kinv(t,Y)
  	#calculate fY = vector [f(A*y1);f(A*y2).....f(A*yt)] (vertical)
  	fY =  acq.calculate_fY(number_of_samples,A,Y,t)
  
  	#exhaustive search in y
  	for i in xrange(0, len(y)):
  		#set new point
  		y_new = y[i]
  	
  		#calculate k vector
  		k_vector = acq.compute_kVector(t,Y,y_new)
  	
  		#calculate mu, sigma
  		temp_mu =  np.dot(np.dot(k_vector,Kinv),fY)
  		temp_sigma = acq.sqexp_kernel(y_new,y_new) - np.dot(np.dot(k_vector,Kinv),k_vector.T)

  		#get candidates through acquisition function
  		candidate =acq.UCB(t,d,temp_mu,temp_sigma)
		candidates.append(candidate)  

	#find the best candidate
	best_index = np.argmax(candidates)	
  	print best_index
  	ybest = y[best_index]
  	print ybest

  
