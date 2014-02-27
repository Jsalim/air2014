#air 2014 group 5. (REMBO)
import create_random_matrix as crm
import chooseBoundedRegion as cbr

D = 64 #number of features?
d = 3 #parameter
regionBound = sqrt(d)
regionBoundStepSize = 0.01

#step 1: generate a random matrix
A = crm.random_matrix(D,d)

#step 2:choose the bounded region
Y = cbr.chooseBoundedRegion(d,regionBound,regionBound,regionBoundStepSize)

#paris - this code is general (doesnt compile)

for (each d we experiment with):
	A = randomMatrix(D,d);
	Y = randomVector(d) #values from -sqrt(d) to sqrt(d)
 	x1,x2 =randomPointsOfY(); #get 2 random points inside Y.
 	ranker1Features =x1*A;
 	ranker2Features = x2*A
 	while(nextQuery):
 		winningRanker = interleave(ranker1,ranker2,someClickModel(),randomQuery()); 
 		updateWeights(winningRanker);



#end paris code




#for (t in range(0,100)):
	#step4

	#step5

