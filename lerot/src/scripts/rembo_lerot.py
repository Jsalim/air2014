# utility imports
import numpy as np
from math import sqrt
from vector_util import get_random_close_point, get_random_point, move_towards_new_point

# REMBO imports
from create_random_matrix import random_matrix2
from choose_bounded_region import choose_bounded_region

# LEROT/L2R imports
from Learning2Rank import Learning2Rank 

# ######### NOTE ######### #
# Mostly based on "interactively 
# optimizing information retrieval 
# systems as a dueling bandit problem"
# ######################## #

# REMBO init
D=64 # features
d=3
min_region_bound = 0 #-sqrt(d)
max_region_bound = 1 #sqrt(d)
region_bound_stepsize = 0.5

# DBGD init, values as in the DBGD paper "interactively optimizing information retrieval systems as a dueling bandit problem"
exploration_stepsize = 1 # BETA in the paper
exploitation_stepsize = 0.01 # GAMMA in the paper

# L2R init (create Learning2Rank class instance)
l2r = Learning2Rank('../../data/NP2004/Fold1/test.txt', D)
	
# ###### REMBO + LEROT ALGORITHM START ####### #

# Step 1
# Generate random matrix
# this mu/sigma might be wrong, I don't know.
mu = 0.5
sigma = 1
A = random_matrix2(D, d, mu, sigma)

# Step 2
# Choose bounded region set: y = N x d matrix (the exhaustive search subset)
y = choose_bounded_region(d, min_region_bound, max_region_bound, region_bound_stepsize)
y = np.array(y)

# start with 1 random sample
Y = get_random_point(y)

# time = amount of queries?
for t in range(1, l2r.get_query_length()):
	# create y_current / y_probe based on y_current
	y_current = Y[len(Y)-1]
	y_probe, altered_dimension = get_random_close_point(y_current, exploration_stepsize, min_region_bound, max_region_bound, d)
	
	# prep for L2R: convert to D-dimensional + create ranker objects
	ranker_current, ranker_probe =  l2r.prepare_rankers(A, y_current, y_probe)
	
	# evaluate
	result = l2r.evaluate(ranker_current, ranker_probe)

	# if ranker_probe better than ranker_current, add new point in Y based on ranker_probe
	if result > 0:
		y_new = move_towards_new_point(y_current, y_probe, altered_dimension, exploitation_stepsize) # we only move towards y_probe a fraction of where we checked as in the paper
		Y = np.append(Y,y_new,0)

		# print "---newY-----"
		# print y_current
		# print y_new

	
	# print "---time-----"
	# print t
	
	# print "---y--------"
	# print y_current
	# print y_probe
	
	# print "---Ay-------"
	# print r_current
	# print r_probe
	
	# print "---result---"
	# print result
	
	print "---Y--------"
	print Y

# comparing when we know relevance values of documents (uses ndcg)
evaluation = evaluation.NdcgEval()
ndcg_result = []
for y_instance in Y:
	ranker_instance =  l2r.prepare_ranker(A, y_instance)
	ndcg_result.append(evaluation.evaluate_all(ranker_instance, queries))
print ndcg_result

# comparing against some other best ranker using regret
regret_result = []
# ranker_best = ranker.ProbabilisticRankingFunction('3', 'random', 64, init=parseRanker('../../data/features64/ranker-02.txt'),sample='sample_unit_sphere')
ranker_best = AbstractRankingFunction(["ranker.model.BM25"], 'first', 3, sample="utils.sample_fixed")
for y_instance in Y:
	ranker_instance =  l2r.prepare_ranker(A, y_instance)
	regret_result.append(l2r.evaluate_multi_query(ranker_instance, ranker_best, 10))
print regret_result