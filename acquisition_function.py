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


# Function to select random point
def select_random_point(y):

  test_set=[]
  idx = np.random.randint(0, len(y))
  test_set.append(y[idx])

  return np.matrix(test_set)


# Function to select test set from bounded box
def select_sample_set(n_test, y):
  test_set = []

  for i in xrange(0, n_test):
    idx = np.random.randint(0, len(y))
    test_set.append(y[idx])

  return np.matrix(test_set)

# Function to return noisy sample
# GP Paper page 3
def sample_training_output(xtrain):
  return f_func(xtrain) + np.random.normal(0, 0.001)

def f_func(x):
  return -(x.sum(1))**2


def get_Kinv(t,Y):
  K = np.zeros([t,t])
  for ind in range(t):
    for jnd in range(t):
      K[ind,jnd] = sqexp_kernel(Y[ind],Y[jnd])
  Kinv = np.linalg.inv(K+0.01*np.eye(t))
  return Kinv


#input: No. of samples output: vector fY
def calculate_fY(number_of_samples,A,Y,t):
  fY = np.zeros([t,1])
  for q in range(0,number_of_samples):
    fY[q]= f_func(np.dot(A,(Y[q].T)).T)
  return fY

def compute_kVector(t,Y,y_new):
  #compute the k vector according to the third paper page 8.
  k_vector = []
  for j in range(0,t):
    k_vector=np.append(k_vector,sqexp_kernel(Y[j],y_new))
  return k_vector
# Function to calculate GP Posterior
# It returns predictive mean and variance
# REMBO paper page 3
def gp_posterior(data,Y, y, A, t, d, number_of_samples):

  mu =[]
  sigma=[]
  candidates=[]

  Kinv = get_Kinv(t,Y)

  fY = np.array()

  for q in range(0,number_of_samples):
    fY[q]=f_func(Y[q])

  for i in xrange(0, len(y)):
    
    y_instance = Y[i]
    y_projected = projection(data,A*y_projected)

    x = np.dot(A,y)
    x = x.T


    #add new line and column to the COV matrix.
    # temp_sigma = np.append(temp_sigma, np.matrix(k_vector), 1)
    # temp = np.append(k_vector, [sqexp_kernel(x, x)], 0)
    # temp_sigma = np.append(temp_sigma, np.matrix(temp), 0)

    # temp = np.matrix(temp)
    # temp = temp.T
    # temp_sigma = np.append(temp_sigma, temp, 0)
    # temp_sigma[:,t+1] = [k_vector[len(k_vector)-1].T]

    #calculate mu and sigma according to the rembo paper.
    # This code is still not working
    temp_mu =  k_vector.T * Kinv * fY
    temp_sigma = sqexp_kernel(y_instance,y_instance) - k_vector.T * Kinv * k_vector

    #call the aqcuisition function here, and find argmax.
    #using temp_mu and temp_sigma
    candidate = UCB(t, d, temp_mu, temp_sigma)
    # print candidate

    candidates.append(candidate)


    mu.append(temp_mu)
    sigma.append(temp_sigma)

  # Find the best candidate
  # print len(candidates)
  # print len(Y)
  best_index = np.argmax(candidates)
  print best_index
  ybest = y[best_index]

  return ybest


#given a point y outside Y, find its projection in Y
def projection(Y,y):
  dist  = 10000
  min_z = Y[1]
  for i in range(0,len(Y)):
    z = Y[i];
    temp_dist = np.linalg.norm(y-z)
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
def UCB(t, d, mu, sigma):
  # ycandidates = []

  # print "=========="
  # print math.sqrt(get_beta(t,d))
  # print "=========="
  # print sigma

  return mu + math.sqrt(get_beta(t, d)) * sigma
  # temp_y = mu[i] + math.sqrt(get_beta(t, d)) * sigma[i]
  # ycandidates.append(temp_y)

  # best_index = np.argmax(ycandidates)
  # return xtest[best_index]

def augment_data(xtrain, fY, xbest, A):
  train = A * xbest.T
  train = train.T

  xtrain = np.concatenate((xtrain, train), 0)
  fY = np.concatenate((fY, f_func(train)), 0)
  return xtrain, fY
