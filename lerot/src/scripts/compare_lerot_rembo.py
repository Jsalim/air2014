# Experiment 2
# Compare REM with original DBGD

import sys, random
import include
import environment, evaluation, query, retrieval_system
import time
import datetime
import numpy as np

# init data, query_samples, d's
train_queries = query.load_queries('../../DATA/NP2004/Fold1/train.txt', 64)
test_queries = query.load_queries('../../DATA/NP2004/Fold1/test.txt', 64)
query_samples = 500 # how many queries we sample

d = 2
k = 1
number_of_evaluation = query_samples / k

# init user model, evaluation methods
user_model = environment.CascadeUserModel('--p_click 0:0.0,1:1 --p_stop 0:0.0,1:0.0')
evaluation = evaluation.NdcgEval()
full_learner = retrieval_system.ListwiseLearningSystem(64,'-w random -c comparison.ProbabilisticInterleave -r ranker.ProbabilisticRankingFunction -s 3 -d 0.1 -a 0.01')

rem_ndcg_evaluation_train = []
full_ndcg_evaluation_train = []
rem_ndcg_evaluation_test = []
full_ndcg_evaluation_test = []

# start k number of runs
for m in range(0, k):
    # for each k, we have different A matrix
    # as mentioned on the REMBO paper
    rem_learner = retrieval_system.ListwiseLearningSystemREMBO(64,d,'-w random -c comparison.ProbabilisticInterleave -r ranker.ProbabilisticRankingFunctionREMBO -s 3 -d 0.1 -a 0.01')

    # start evaluation
    for i in range(0,number_of_evaluation):
        q = train_queries[random.choice(train_queries.keys())]

        l_rem = rem_learner.get_ranked_list(q)
        l_full = full_learner.get_ranked_list(q)

        c_rem = user_model.get_clicks(l_rem, q.get_labels())
        c_full = user_model.get_clicks(l_full, q.get_labels())

        s_rem = rem_learner.update_solution(c_rem)
        s_full = full_learner.update_solution(c_full)

        rem_ndcg_evaluation_train.append(evaluation.evaluate_all(s_rem, train_queries))
        full_ndcg_evaluation_train.append(evaluation.evaluate_all(s_full, train_queries))
        # rem_ndcg_evaluation_test.append(evaluation.evaluate_all(s_rem, test_queries))
        # full_ndcg_evaluation_test.append(evaluation.evaluate_all(s_full, test_queries))

# write the result to file
timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')
f = open("../../output/experiment2/" + timestamp + "_train.txt", "w")
f.write("k: %s" % str(k) + "\n")
f.write("d: %s" % str(d) + "\n")
f.write("rem_ndcg_evaluation: %s" % str(rem_ndcg_evaluation_train) + "\n")
f.write("full_ndcg_evaluation: %s" % str(full_ndcg_evaluation_train))
f.close()
# f2 = open("../../output/experiment2/" + timestamp + "_test.txt", "w")
# f2.write("k: %s" % str(k) + "\n")
# f2.write("d: %s" % str(d) + "\n")
# f2.write("rem_ndcg_evaluation: %s" % str(rem_ndcg_evaluation_test) + "\n")
# f2.write("full_ndcg_evaluation: %s" % str(full_ndcg_evaluation_test))
# f2.close()