import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
import imagebase
import neighbour

def postiter(image,x,y):
    tempimage = image
    for i in range(x):
        for j in range(y):
            sum = 0
            for ne in neighbour.neighbours(i,j,x,y,8):
                sum += image[ne]
            mean = sum / len(neighbour.neighbours(i,j,x,y,8))
            if(sum > 0.5):
                tempimage[i,j] = 1
            else:
                tempimage[i,j] = 0
    image = tempimage
    return image

def denoise(image):
    x = image.shape[0]
    y = image.shape[1]
    for t in range(1):
        image = postiter(image,x,y)

    return image

prop = 0.7
varSigma = 0.1

im = imread('chromegray.png')
#im[i] = [0..1]
im = im/255


fig = plt.figure()
ax = fig.add_subplot(231)
ax.imshow(im,cmap='gray')
imG = imagebase.add_gaussian_noise(im,prop,varSigma)
ax2 = fig.add_subplot(232)
ax2.imshow(imG,cmap='gray')
imS = imagebase.add_saltnpeppar_noise(im,prop)
ax3 = fig.add_subplot(233)
ax3.imshow(imS,cmap='gray')
denG = denoise(imG)
denS = denoise(imS)
ax4 = fig.add_subplot(235)
ax4.imshow(imG,cmap='gray')
ax5 = fig.add_subplot(236)
ax5.imshow(imS,cmap='gray')
