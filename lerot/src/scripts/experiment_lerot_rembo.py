# Experiment 1
# Get optimum configuration for REM (number of k, and d)

import sys, random
import include
import environment, evaluation, query, retrieval_system
import time
import datetime

# init data, query_samples, d's
queries = query.load_queries('../../DATA/NP2004/Fold1/train.txt', 64)
test_queries = query.load_queries('../../DATA/NP2004/Fold1/test.txt', 64)
query_samples = 30 # how many queries we sample

d_array = [2,3,4,5,6]
k_array = [1,2,5,10]

# init user model, evaluation methods
user_model = environment.CascadeUserModel('--p_click 0:0.0,1:1 --p_stop 0:0.0,1:0.0')
evaluation = evaluation.NdcgEval()

# calculate using the lower dimensional slice(s) in d_array
rem_ndcg_result = [[0 for i in range(len(d_array))] for j in range(len(k_array))]

for n in range(0, len(k_array)):
    # number of evaluation for each k
    # make sure the number of total evaluation are equal for all k
    number_of_evaluation = query_samples / k_array[n]
    k = k_array[n]

    for idx in range(0,len(d_array)):
        d = d_array[idx]
        temp_ndcg_evaluation = []

        # start k number of runs
        for m in range(0, k):
            # for each k, we have different A matrix
            # as mentioned on the REMBO paper
            rem_learner = retrieval_system.ListwiseLearningSystemREMBO(64,d,'-w random -c comparison.ProbabilisticInterleave -r ranker.ProbabilisticRankingFunctionREMBO -s 3 -d 0.1 -a 0.01')

            # start evaluation
            for i in range(0,number_of_evaluation):
                q = queries[random.choice(queries.keys())]
                l = rem_learner.get_ranked_list(q)
                c = user_model.get_clicks(l, q.get_labels())
                s = rem_learner.update_solution(c)
                temp_ndcg_evaluation.append(evaluation.evaluate_all(s, test_queries))

        # calculate average ndcg for all evaluation
        # rem_ndcg_result[n][idx] = sum(temp_ndcg_evaluation) / float(len(temp_ndcg_evaluation))
        rem_ndcg_result[n][idx] = temp_ndcg_evaluation

# write the result to file
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
f = open("../../output/experiment1/" + timestamp + ".txt", "w")
f.write("k: %s" % str(k_array) + "\n")
f.write("d: %s" % str(d_array) + "\n")
f.write("ndcg: %s" % str(rem_ndcg_result))
f.close()
