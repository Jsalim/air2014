import random
import include
import ranker, environment, comparison, query
import numpy as np

# class which takes care of the L2R/Lerot part of the REMBO algorithm.
# it is the only component within the entire algorithm which communicates 
# with all the components in Lerot
class Learning2Rank:
	# init function. Simply inits the interleave model, user model, and queries
	def __init__(self, data_path, features):
		self.bi = comparison.BalancedInterleave()
		self.user_model = environment.CascadeUserModel('--p_click 0:0.0,1:1 --p_stop 0:0.0,1:0.0')	
		self.queries = query.load_queries(data_path, features)
	
	# returns the amount of queries there are
	def get_query_length(self):
		return len(self.queries)
	
	# returns two initialized ranker objects
	def prepare_rankers (self, A, y1, y2):
		# matrix multiplication with random matrix A
		Ay1 = np.dot(A,(y1.T)).T
		Ay2 = np.dot(A,(y2.T)).T
		
		# convert to string
		string_Ay1 = ', '.join(map(str, np.squeeze(np.asarray(Ay1))))
		string_Ay2 = ', '.join(map(str, np.squeeze(np.asarray(Ay2))))
		
		# create ranker objects
		r1 = ranker.ProbabilisticRankingFunction('3', 'random', 64, init=string_Ay1, sample='sample_unit_sphere')
		r2 = ranker.ProbabilisticRankingFunction('3', 'random', 64, init=string_Ay2, sample='sample_unit_sphere')
		return r1, r2
		
	# returns a single initialized ranker object
	def prepare_ranker (self, A, y):
		# matrix multiplication with random matrix A
		Ay = np.dot(A,(y.T)).T
		
		# convert to string
		string_Ay = ', '.join(map(str, np.squeeze(np.asarray(Ay))))
		
		# create ranker objects
		r = ranker.ProbabilisticRankingFunction('3', 'random', 64, init=string_Ay, sample='sample_unit_sphere')
		return r
	
	# strips ranker files to their bare necessities 
	def parse_ranker (path): 
		f = open(path,'r') 
		str = f.read().lstrip('final_weights: [')
		str = str.rstrip('] \n') 
		f.close() 
		return str	
		
	# evalutes two rankers, not unlike as was done in the assignment
	def evaluate(self, ranker1, ranker2):
		q = self.queries[random.choice(self.queries.keys())]
		list, context = self.bi.interleave(ranker1, ranker2, q, 10)
		clicks = self.user_model.get_clicks(list,q.get_labels())
		result = self.bi.infer_outcome(list,context,clicks,q)
		return result
		
	# evaluate two rankers using all queries as opposed to one
	def evaluate_multi_query(self, ranker1, ranker2, iters):
		for i in range(0,iters):
			result = evaluate(ranker1, ranker2)
			if result < 0:
				
		return result