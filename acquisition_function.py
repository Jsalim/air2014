import numpy as np
import scipy.spatial.distance as sp
import math

# see REMBO paper page 5 definition 4
# l = length scale > 0
# defined at first run, should be tuned

l = 1


def get_beta(t,d):
    delta = 0.01
    #a and b: to find out! it is a constant, see therom 2 of
    #gaussian process paper.
    a = 1
    b = 1
    r = 1  #theorem 2, page 5, has something to do with D
    t = t+1
    beta = 2*np.log(t**2*r*math.pi**2/(3*delta)) + 2*d*np.log(t**2*d*b*r*math.sqrt(np.log(4*d*a/delta)))
    return beta



# Function to select test set from bounded box
def select_test_set(n_test, Y):
  test_set = []

  for i in xrange(0, n_test):
    idx = np.random.randint(0, len(Y))
    test_set.append(Y[idx])

  return np.matrix(test_set)

# Function to return noisy sample
# GP Paper page 3
def sample_training_output(xtrain):
  return f_func(xtrain) + np.random.normal(0, 0.001)

def f_func(x):
  return x.sum(1)

# Function to calculate GP Posterior
# It returns predictive mean and variance
# REMBO paper page 3
def gp_posterior(data_old, sigma_old, Y, ytrain, sigma_0, n_test, A):

  mu =[]
  sigma=[]

  for i in xrange(0, len(Y)):
    test = A * Y[i].T
    test = test.T
    train = data_old

    temp_mu 

    temp_mu =  ????* np.linalg.inv(sigma_old) * ytrain
    temp_sigma = sqexp_kernel(test, test) - (sqexp_kernel(test, train) * sqexp_kernel(train, train).I * sqexp_kernel(train, test))
    
      #call the aqcuisition function here, and find argmax.
      #using temp_§mu and temp_sigma

    mu.append(temp_mu)
    sigma.append(temp_sigma)

  return mu, sigma


# Function squared exponential kernel a.k.a radial basis function kernel
# REMBO paper page 5
def sqexp_kernel(y1, y2):
  # length scale
  # should be tuned
  l = 1

  # we are using squared euclidan distance
  # http://mlg.eng.cam.ac.uk/duvenaud/cookbook/index.html
  # http://en.wikipedia.org/wiki/Radial_basis_function_kernel
  distance = sp.cdist(y1, y2, 'sqeuclidean')
  k = np.exp(-(distance/2*(l**2)))
  return np.matrix(k)

# Acquisition Function
# GP-UCB Algorithm
# GP Paper page 4
def gp_optimize(xtest, t, d, mu, sigma, n_test):
  t = t + 1
  ycandidates = []

  for i in xrange(0, n_test):
    temp_y = mu[i] + math.sqrt(get_beta(t, d)) * sigma[i]
    ycandidates.append(temp_y)
  best_index = np.argmax(ycandidates)
  return xtest[best_index]

def augment_data(xtrain, ytrain, xbest, A):
  train = A * xbest.T
  train = train.T

  xtrain = np.concatenate((xtrain, train), 0)
  ytrain = np.concatenate((ytrain, f_func(train)), 0)
  return xtrain, ytrain
