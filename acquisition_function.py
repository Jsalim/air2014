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
def gp_posterior(data_old, sigma_old, Y, ytrain, A, t, d):

  mu =[]
  sigma=[]
  candidates=[]

  for i in xrange(0, len(Y)):
    row = np.matrix(Y[i])
    test = A * row.T
    test = test.T
    train = data_old


    #compute the k vector according to the third paper page 8.
    temp_sigma = sigma_old
    k_vector = []
    for j in range(0,t+1):
      k_vector.append(sqexp_kernel(data_old[j],test))


    #add new line and column to the COV matrix.
    # This code is only working for t = 0
    # Need to be rewriting
    temp_sigma = np.append(temp_sigma, np.matrix(k_vector), 0)
    # temp_sigma[t+1,t+1] =  sqexp_kernel(test,test)
    temp = np.append(k_vector, [sqexp_kernel(test, test)], 0)
    temp_sigma = np.append(temp_sigma, [k_vector, [sqexp_kernel(test, test)]], 1)
    # temp = np.matrix(temp)
    # temp = temp.T
    # temp_sigma = np.append(temp_sigma, temp, 0)
    print temp_sigma
    # temp_sigma[:,t+1] = [k_vector[len(k_vector)-1].T]




    #calculate mu and sigma according to the rembo paper.
    # This code is still not working
    temp_mu =  k_vector.T* np.linalg.inv(sigma_old) * ytrain
    temp_sigma = sqexp_kernel(test, test) - k_vector.T * np.linalg.inv(sigma_old) * k_vector

    #call the aqcuisition function here, and find argmax.
    #using temp_mu and temp_sigma
    candidate = gp_optimize(test, t, d, temp_mu, temp_sigma)
    candidates.append(candidate)

    # Find the best candidate
    best_index = np.argmax(ycandidates)
    ybest = test[best_index]

    mu.append(temp_mu)
    sigma.append(temp_sigma)

  return mu, sigma, ybest


#given a point y outside Y, find its projection in Y
def projection(Y,y):
  dist  = 10000
  min_z = Y[1]
  for i in range(0,len(Y)):
    z = Y[i];
    temp_dist = numpy.linalg.norm(y-z)
    if temp_dist<dist:
      dist=temp_dist
      min_z=Y[i]
  return min_z




# Function squared exponential kernel a.k.a radial basis function kernel
# REMBO paper page 5
def sqexp_kernel(y1, y2):
  # length scale
  # should be tuned
  l = 1

  # we are using squared euclidan distance
  # http://mlg.eng.cam.ac.uk/duvenaud/cookbook/index.html
  # http://en.wikipedia.org/wiki/Radial_basis_function_kernel
  distance = sp.euclidean(y1, y2)
  k = np.exp(-(distance/2*(l**2)))
  # return np.matrix(k)
  return k

# Acquisition Function
# GP-UCB Algorithm
# GP Paper page 4
def gp_optimize(t, d, mu, sigma):
  t = t + 1
  # ycandidates = []

  return mu[i] + math.sqrt(get_beta(t, d)) * sigma[i]
  # temp_y = mu[i] + math.sqrt(get_beta(t, d)) * sigma[i]
  # ycandidates.append(temp_y)

  # best_index = np.argmax(ycandidates)
  # return xtest[best_index]

def augment_data(xtrain, ytrain, xbest, A):
  train = A * xbest.T
  train = train.T

  xtrain = np.concatenate((xtrain, train), 0)
  ytrain = np.concatenate((ytrain, f_func(train)), 0)
  return xtrain, ytrain
