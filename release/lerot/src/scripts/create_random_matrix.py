import numpy as np

# Random matrix D x d with normal distribution
# with mean = 0, and standar deviation = 1
# REMBO paper page 6
def random_matrix(D, d):
  return np.random.normal(0, 1, (D, d))
  
def random_matrix2(D,d,mu,sigma):
  return np.random.normal(mu,sigma, (D, d))
