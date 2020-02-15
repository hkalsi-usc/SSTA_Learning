import matplotlib.pyplot as plt
import numpy as np
import math
import random

class norml_plt:
    def __init__(self, ran):
        self.mu = np.mean(ran)
        self.sigma = np.std(ran)
        self.size = len(ran)
        self.ran = ran
    def curve(self, color, resolution_bits):
        n, bins, patches = plt.hist(self.ran, int(self.size/resolution_bits), density = 1)
        plt.plot(bins, (1 / (np.sqrt(2 * np.pi) * self.sigma)) *
            np.exp( - (bins - self.mu)**2 / (2 * self.sigma**2) ), linewidth=2, color= color)
    #plt.show()

plt.figure()
plt.subplot2grid((2, 2), (0, 0), colspan=2)
mu1 = 5
sigma1 = 3
size1 = 10
x = np.random.normal(mu1, sigma1, size1)
print(x)
p1 = norml_plt(x)
plt.subplot2grid((2, 2), (0, 0))
p1.curve('g',3)

mu2 = 5
sigma2 = 2
size2 = 8
y = np.random.normal(mu2, sigma2, size2)
print(y)
p2 = norml_plt(y)
plt.subplot2grid((2, 2), (0, 1))
p2.curve('r',1)



z = np.convolve(x,y)
print(z)
pr = norml_plt(z)
plt.subplot2grid((2, 2), (1, 0),colspan=2)
pr.curve('b',3)


"""
p1 = norml_plt(x)
p1.curve('b')
"""
plt.show()
