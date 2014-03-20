import sys, random
import include
import environment, evaluation, query, retrieval_system

# init data, query_samples, d's
queries = query.load_queries('../../data/NP2004/Fold1/test.txt', 64)
query_samples = 10 # how many queries we sample
d_array = [3,4,5,6]

# init user model, evaluation methods
user_model = environment.CascadeUserModel('--p_click 0:0.0,1:1 --p_stop 0:0.0,1:0.0')
evaluation = evaluation.NdcgEval()

# calculate using the full 64 dimensions
full_learner = retrieval_system.ListwiseLearningSystem(64,'-w random -c comparison.ProbabilisticInterleave -r ranker.ProbabilisticRankingFunction -s 3 -d 0.1 -a 0.01')
full_ndcg_result = []
for i in range(0,query_samples):
    q = queries[random.choice(queries.keys())]
    l = full_learner.get_ranked_list(q)
    c = user_model.get_clicks(l, q.get_labels())
    s = full_learner.update_solution(c)
    full_ndcg_result.append( evaluation.evaluate_all(s, queries) )
full_ranker = full_learner.get_solution()

# calculate using the lower dimensional slice(s) in d_array
rem_ndcg_result = [[0 for i in range(len(d_array))] for j in range(query_samples)]
rem_ranker = []
for idx in range(0,len(d_array)):
    d = d_array[idx]
    rem_learner = retrieval_system.ListwiseLearningSystemREMBO(64,d,'-w random -c comparison.ProbabilisticInterleave -r ranker.ProbabilisticRankingFunctionREMBO -s 3 -d 0.1 -a 0.01')
    for i in range(0,query_samples):
        q = queries[random.choice(queries.keys())]
        l = rem_learner.get_ranked_list(q)
        c = user_model.get_clicks(l, q.get_labels())
        s = rem_learner.update_solution(c)
        rem_ndcg_result[i][idx] = evaluation.evaluate_all(s, queries) 
    rem_ranker.append( rem_learner.get_solution() )

# TODO : 
# - wrap the current for-loops in another one so that we use the average of a number of runs, so that we're sure we get some steady results (this is the K in the REMBO paper)
# - plot NDCGs of ALL rem_ranker and full_ranker_weight
# - compare the rem_ranker using code similar to the assignment we did (interleave/infer_outcome), 
# - then compare the best one against full_ranker using a similar scheme

print max(full_ndcg_result) # prints max ndcg of the full ranker
print full_ndcg_result[-1] # prints last ndcg of the full ranker
print max(rem_ndcg_result[-1]) # prints max ndcg of the last rem_ranker we tested
print rem_ndcg_result[-1][-1] # prints the last ndcg of the last rem_ranker we tested
