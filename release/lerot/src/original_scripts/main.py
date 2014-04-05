# try:
    # from include import *
# except:
    # pass
# from experiment import GenericExperiment

# if __name__ == "__main__":
    # experiment = GenericExperiment()
    # experiment.run()
	
import unittest
import sys
import os
import cStringIO

sys.path.insert(0, os.path.abspath('..\python'))

import query
from numpy import array
sys.path.insert(0, os.path.abspath('C:\Users\Reinier\lerot\src\python\retrieval_system'))
# from PairwiseLearningSystem import PairwiseLearningSystem

test = PairwiseLearningSystem()
test_num_features = 64
f = open('C:\Users\Reinier\lerot\DATA\NP2004\Fold1\train.txt', 'r')
print f

