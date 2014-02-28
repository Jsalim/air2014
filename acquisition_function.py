import numpy as np
import math

# see REMBO paper page 5 definition 4
# l = length scale > 0
# defined at first run, should be tuned
l = 1

def get_kT():
            

def get_KT(y1, y2):
    #y1 = yt, y2 = yt+1 ; could be the other way around
    return np.exp((-(np.abs(np.subtract(y1,y2)))**2)/(2*l**2))

def get_yt():


def get_beta(t,d):
    delta = 0.01
    #a and b: to find out! it is a constant, see therom 2 of
    #gaussian process paper.
    a = 1  
    b = 1  
    r = 1  #theorem 2, page 5, has something to do with D
    beta = 2*np.log(t**2*r*math.pi**2/(3*delta)) + 2*d*np.log(t**2*d*b*r*math.sqrt(numpy.log(4*d*a/delta)))

def get_sigma():
    return get_kT()**2

def get_mu():
    get_kT() * (get_KT + sigma**2)^(-1)* get_yt() #*y_t
    return #something

def acquisition(Y,A,d):
    # each function that is called should still get the proper input,
    # we don't know what that should be yet
    yt = np.argmax(get_mu() + math.sqrt(get_beta())*get_sigma())

    return y
