import sys, os, cStringIO, random, operator
try:
    import include
except:
    pass
import ranker, environment, comparison, query

def perform():
	# init
	test_num_features = 64
	queries = query.load_queries('../../data/NP2004/Fold1/test.txt', 64)
	bi = comparison.BalancedInterleave()
	user_model = environment.CascadeUserModel('--p_click 0:0.0,1:1 --p_stop 0:0.0,1:0.0')	
	
	# make rankers
	rankers = []
	for i in range(0,5):
		rankers.append(ranker.ProbabilisticRankingFunction('3', 'random', 64, init=parseRanker('../../data/features64/ranker-0'+str(i)+'.txt'),sample='sample_unit_sphere'))

	# main loop
	for N in [100,1000,10000]:
		pref_matrix = [[0 for x in xrange(5)] for x in xrange(5)] 
		for iter in range(0,N):
			q = queries[random.choice(queries.keys())]
			for i in range(0,5):
				for j in range (0,5):
					if i!=j:
						list, context = bi.interleave(rankers[i], rankers[j], q, 10)
						clicks = user_model.get_clicks(list,q.get_labels())
						result = bi.infer_outcome(list,context,clicks,q)
						if result < 0:
							pref_matrix[i][j] += 1
					else:
						pref_matrix[i][j] = 0.50
		pref_matrix = generateProbabilityMatrix(pref_matrix,N)
		printMatrix(pref_matrix)
		print 'Best ranker is ' + '0' + str(getBestRanker(pref_matrix)) + ' (N = ' + str(N) + ').'
	print 'done!'

# assist methods

# strips ranker files to their bare necessities
def parseRanker (path):
	f = open(path,'r')
	str = f.read().lstrip('final_weights: [')
	str = str.rstrip('] \n')
	f.close()
	return str
	
# prints the matrix in a pretty way
def printMatrix(m):
	print ' ',
	for i in range(len(m[1])):
		print str(i)+'   ',
	print
	for i, element in enumerate(m):
		print i, " ".join([str(y) for y in element])

# creates probability matrix
def generateProbabilityMatrix(m,N):
	result = [[0 for x in xrange(5)] for x in xrange(5)] 
	for i in range(0,5):
		for j in range (0,5): 
			if i!=j:
				if float(m[i][j]) + float(m[j][i]) != 0:
					result[i][j] = '%.2f' % (float(m[i][j]) / float((m[i][j])+float(m[j][i])))
				else:
					result[i][j] = '0.00'
			else:
				result[i][j] = '0.50'
	return result

# calculates best ranker
def getBestRanker(m):
	a = []
	for i in range(0,5):
		sum = 0
		for j in range (0,5):
			sum = sum + float(m[i][j])
		a.append(sum)
	best_idx, best_val = max(enumerate(a), key=operator.itemgetter(1))
	return best_idx

if __name__ == '__main__':
	perform()