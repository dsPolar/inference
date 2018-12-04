import numpy as np
import imageio
import matplotlib.pyplot as plt
from scipy.misc import imread
import imagebase
import neighbour
import math


def to_1d(rgb_image):
    ret = []
    x_dim, y_dim, z = rgb_image.shape
    for i in range(0, x_dim):
        ret_x = []
        for j in range(0, y_dim):
            ret_x.append(round(np.sum(rgb_image[i,j])/3, 1))
        ret.append(ret_x)
    return np.array(ret)

def sigm(v):
    return (1 / (1 + np.exp(-v)))

#takes an image point value a 0..1
#takes a check value v 0..1
#returns Likelihood that a yields v
def p(a,v):
    res = 0
    thresh = v /2
    if(a >= thresh):
        res = 1
    else:
        res = 0
    return res

def newp(a,v):
    return (a + (v-a)/2)


def l(image,i,j,x,y,v):
    res = 0
    sum = 0
    for ne in neighbour.neighbours(i,j,x,y,4):
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
    m = np.zeros((iter+1,x,y),order='C')
    mu = np.ones((iter+1,x,y),order='C')

    q = np.zeros((x,y),order='C')
    #image is latent

    #correct access is m[t,x,y]
    for t in range(iter):
        #[ii][jj] pair comprises the singular i iterator used in CW specification
        print(t)
        for ii in range(x):
            for jj in range(y):
                sum = 0
                for ne in neighbour.neighbours(ii,jj,x,y,8):
                    #Add the -1..1 product value of neighbour with current, modified by the mu value for neighbour
                    sum += (2*image[ne] -1) * (2*image[ii][jj] -1) * mu[t,ne[0],ne[1]]

                m[t+1,ii,jj] = sum
                #mu = tanh(m_i^tau + 1/2(L_i(1) - L_i(-1)))
                mu[t+1,ii,jj] = math.tanh(m[t,ii,jj] + 1/2*(newp(image[ii][jj],1) - newp(image[ii][jj],0)))

    qProb = 1
    for i in range(x):
        for j in range(y):
            q[i,j] = sigm(2*(m[iter,i,j] + 1/2*(newp(image[i][j],1) - newp(image[i][j],0))))
            qProb *= q[i,j]

    #returns variational distro and mu
    return (q,mu,m)


prop = 0.7
varSigma = 0.1

im = to_1d(imageio.imread('pug.jpg'))
#im[i] = [0..1]
im = im/255
x = im.shape[0]
y = im.shape[1]

fig = plt.figure()
ax = fig.add_subplot(231)
ax.imshow(im,cmap='gray')
imG = imagebase.add_gaussian_noise(im,prop,varSigma)
ax2 = fig.add_subplot(232)
ax2.imshow(imG,cmap='gray')
imS = imagebase.add_saltnpeppar_noise(im,prop)
ax3 = fig.add_subplot(233)
ax3.imshow(imS,cmap='gray')

qS,muS,mS = mfvb(imS,7)
qG,muG,mG = mfvb(imG,7)


for i in range(x):
    for j in range(y):
        if(qS[i,j] < 0.5):
            if(imS[i,j] >= 0.5):
                imS[i,j] = newp(imS[i,j],0)
            else:
                imS[i,j] = newp(imS[i,j],1)
        else:
            if(imS[i,j] >= 0.5):
                imS[i,j] = newp(imS[i,j],1)
            else:
                imS[i,j] = newp(imS[i,j],0)

for i in range(x):
    for j in range(y):
        if(qG[i,j] < 0.5):
            if(imG[i,j] >= 0.5):
                imG[i,j] = newp(imG[i,j],0)
            else:
                imG[i,j] = newp(imG[i,j],1)
        else:
            if(imG[i,j] >= 0.5):
                imG[i,j] = newp(imG[i,j],1)
            else:
                imG[i,j] = newp(imG[i,j],0)


ax4 = fig.add_subplot(235)
ax4.imshow(imG,cmap='gray')
ax5 = fig.add_subplot(236)
ax5.imshow(imS,cmap='gray')

plt.savefig("vb.png")
