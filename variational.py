import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.misc import imread
import imagebase
import neighbour
import math


def sigm(v):
    return (1 / (1 + np.exp(-v)))

#takes an image point value a 0..1
#takes a check value v -1..1
#returns Likelihood that a yields v
def p(a,v):
    res = 0
    val = (v + 1)/2
    if(a == val):
        res = 1
    else:
        res = 0
    return res

def l(image,i,j,x,y,v):
    res = 0
    sum = 0
    for ne in neighbour.neighbours(i,j,x,y,8):
        sum += p(image[ne],image[i][j])
    swap = sum/8
    if(image[i][j] == v):
        res = 1 - swap
    else:
        res = 0 + swap
    return res

def mfvb(image,iter):

    x = image.shape[0]
    y = image.shape[1]
    #Initialise the variational distributions
    #Initialise M
    m = np.zeros((x,y,iter+1),order='C')
    mu = np.zeros((x,y,iter+1),order='C')
    q = np.zeros((x,y),order='C')
    #image is latent

    for t in range(iter):
        #[ii][jj] pair comprises the singular i iterator used in CW specification
        for ii in range(x):
            for jj in range(y):
                sum = 0
                for ne in neighbour.neighbours(ii,jj,x,y,8):
                    #Add the -1..1 product value of neighbour with current, modified by the mu value for neighbour
                    sum += (2*image[ne] -1) * (2*image[ii][jj] -1) * mu[ne[0],ne[1],t]

                m[ii,jj,t+1] = 0
                mu[ii,jj,t+1] = math.tanh(m[ii,jj,t] + 1/2*(p(image[ii][jj],1) - p(image[ii][jj],-1)))

    qProb = 1
    for i in range(x):
        for j in range(y):
            q[i,j] = sigm(2*(m[ii,jj,iter] + 1/2*(p(image[ii][jj],1) - p(image[ii][jj],-1))))
            qProb *= q[i,j]

    #returns q(x == image)
    return qProb


prop = 0.7
varSigma = 0.1

im = imageio.imread('chromegray.png')
#im[i] = [0..1]
im = im/255

imG = imagebase.add_gaussian_noise(im,prop,varSigma)
imS = imagebase.add_saltnpeppar_noise(im,prop)
posterior = 0
posterior = mfvb(im,20)
