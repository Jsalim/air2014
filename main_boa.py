# Bayesian Optimization

import numpy as np
import math

import acquisition_function as acq

D = 10 # number of features
n_training = 10
n_test = 5
max_iter = 100 # maximum number of iterations
sigma_0 = 0.001


# Sample training input and ouput
ytrain = np.matrix(np.random.uniform(-5, 5, (n_training, D)))
fytrain = acq.sample_training_output(ytrain)

# Step 3 - 6
for t in range(0, max_iter):
  # Set of points to be tested
  ytest = np.matrix(np.random.uniform(-5, 5, (n_test, D)))

  # Get mu and sigma
  mu, sigma = acq.gp_posterior(ytrain, ytest, fytrain, sigma_0, n_test)

  # Find ybest
  ybest = acq.gp_optimize(ytest, t, D, mu, sigma, n_test)

  # Augment the data
  ytrain, fytrain = acq.augment_data(ytrain, fytrain, ybest)

  print ybest
