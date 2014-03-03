import numpy as np

# Random matrix D x d with normal distribution
# with mean = 0, and standar deviation = 1
# REMBO paper page 6
def random_matrix(D, d):
  return np.matrix(np.random.normal(0, 1, (D, d)))
