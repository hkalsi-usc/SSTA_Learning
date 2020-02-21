from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import scipy
from scipy import stats
from scipy.stats import norm
import seaborn as sns

mu1 = 1
sigma1 = 1
xsize = 1000
x = np.random.normal(mu1, sigma1, xsize)
x.sort()
xcdf = scipy.stats.norm.cdf(x)
xpdf = scipy.stats.norm.pdf(x,loc=mu1)

#plt.figure()
#g = sns.lineplot(x=x, y = xpdf, color = "blue")


mu2 = 1
sigma2 = 1
ysize = 1000
y = np.random.normal(mu2, sigma2, ysize)
y.sort()
ycdf = scipy.stats.norm.cdf(y)
ypdf = scipy.stats.norm.pdf(y,loc=mu2)
#g = sns.lineplot(x=y, y=ypdf, color = "yellow")

#print("size of x ", len(x))

z = []
zpdf = []

def max(A, B):
    YPDF = 0
    YCDF = 0
    XPDF = 0 
    XCDF = 0
    for i in range (xsize-1):
        z.append(x[i])
        for j in range (ysize-1):
            if(z[i]>=y[j]):
                YCDF = ycdf[j]
                YPDF = ypdf[j]  
        if(z[i]>x[xsize-1]):
            YPDF = 0
            YCDF = 1
        elif(z[i]<x[0]):
            YPDF = 0
            YCDF = 0
        zpdf.append(xcdf[i]*YPDF+YCDF*xpdf[i])

    for i in range (ysize-1):
        temp = y[i]
        if(temp<[x[0]]):
            i = i
        else:
            z.append(y[i])
            for j in range (xsize-1):
                if(temp>=x[j]):
                    XCDF = xcdf[j]
                    XPDF = xpdf[j]  
            if(temp>x[xsize-1]):
                XPDF = 0
                XCDF = 1
            zpdf.append(XCDF*ypdf[i]+ycdf[i]*XPDF)
    #g = sns.lineplot(x=z, y=zpdf, color = "red")
    #print("The mean of maximum between A and B is: ", np.mean(z))

max(x,y)
plt.figure()
g = sns.lineplot(x=x, y = xpdf, color = "blue")
g = sns.lineplot(x=y, y=ypdf, color = "yellow")
g = sns.lineplot(x=z, y=zpdf, color = "red")
plt.show()