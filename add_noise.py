import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
from skimage import color

def add_gaussian_noise(im, prop, varSigma):
    N = int(np.round(np.prod(im.shape) * prop))

    index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N], im.shape)
    e = varSigma * np.random.randn(np.prod(im.shape)).reshape(im.shape)
    im2 = np.copy(im)
    im2[index] += e[index]

    return im2

def add_saltnpepper_noise(im, prop):
    N = int(np.round(np.prod(im.shape) * prop))
    index = np.unravel_index(np.random.permutation(np.prod(im.shape))[1:N], im.shape)
    im2 = np.copy(im)
    im2[index] = 1-im2[index]

    return im2

# Plot Gaussian noise and Salt and pepper noise, save images
def add_noise(image_path, prop, varSigma):
    im =imread(image_path)/255
    im = color.rgb2gray(im)
    for i in range(0, im.shape[0]):
        for j in range(0, im.shape[1]):
            im[i, j] = round(im[i,j])
    im_noise = add_saltnpepper_noise(im, prop)
    fig = plt.figure()
    ax = fig.add_subplot(131)
    plt.imsave('assets/im_noise.jpg', im_noise, cmap='gray')
    return im_noise
