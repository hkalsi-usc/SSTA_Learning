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
asize = 1000
a = np.random.normal(mu1, sigma1, asize)
a.sort()
acdf = scipy.stats.norm.cdf(a)
apdf = scipy.stats.norm.pdf(a,loc=mu1)

#plt.figure()
#g = sns.lineplot(x=a, y = apdf, color = "blue")


mu2 = 1
sigma2 = 1
bsize = 1000
b = np.random.normal(mu2, sigma2, bsize)
b.sort()
bcdf = scipy.stats.norm.cdf(b)
bpdf = scipy.stats.norm.pdf(b,loc=mu2)
#g = sns.lineplot(x=y, y=ypdf, color = "yellow")

mu3 = 1
sigma3 = 1
csize = 1000
c = np.random.normal(mu3, sigma3, csize)
c.sort()
ccdf = scipy.stats.norm.cdf(c)
cpdf = scipy.stats.norm.pdf(c,loc=mu3)
#g = sns.lineplot(x=y, y=ypdf, color = "yellow")

#print("size of a ", len(a))

def max(x, xcdf, xpdf, y, ycdf, ypdf):

    #print("max of x in max: ", np.max(x))
    YPDF = 0
    YCDF = 0
    XPDF = 0 
    XCDF = 0
    z = []
    zpdf = []
    for i in range (len(x)-1):
        z.append(x[i])
        for j in range (len(y)-1):
            if(z[i]>=y[j]):
                YCDF = ycdf[j]
                YPDF = ypdf[j]  
        if(z[i]>x[len(x)-1]):
            YPDF = 0
            YCDF = 1
        elif(z[i]<x[0]):
            YPDF = 0
            YCDF = 0
        zpdf.append(xcdf[i]*YPDF+YCDF*xpdf[i])

    for i in range (len(y)-1):
        temp = y[i]
        if(temp<[x[0]]):
            i = i
        else:
            z.append(y[i])
            for j in range (len(x)-1):
                if(temp>=x[j]):
                    XCDF = xcdf[j]
                    XPDF = xpdf[j]  
            if(temp>x[len(x)-1]):
                XPDF = 0
                XCDF = 1
            zpdf.append(XCDF*ypdf[i]+ycdf[i]*XPDF)
    return (z, zpdf)

def sumofmax(a, acdf, apdf, b, bcdf, bpdf, c, cdf, cpdf):
    z, zpdf = max(a, acdf, apdf, b, bcdf, bpdf)
    SOM = np.convolve(c, z)
    return (SOM, z, zpdf)

def maxofsum(a, b, c):
    sum1 = np.convolve(a, c)
    sum2 = np.convolve(b, c)
    sum1.sort()
    sum2.sort()
    print("This is sum1 after sort: ", sum1)
    sum1cdf = scipy.stats.norm.cdf(sum1)
    sum1pdf = scipy.stats.norm.pdf(sum1,loc=np.mean(sum1))
    sum2cdf = scipy.stats.norm.cdf(sum2)
    sum2pdf = scipy.stats.norm.pdf(sum2,loc=np.mean(sum2))
    print("This is sum1cdf after sort: ", sum1cdf)
    result, resultpdf = max(sum1, sum1cdf, sum1pdf, sum2, sum2cdf, sum2pdf)
    return (result,resultpdf, sum1, sum2)

#z, zpdf = max(a,b)

SOM, z, zpdf = sumofmax(a, acdf, apdf, b, bcdf, bpdf, c, ccdf, cpdf)
MOS, MOSpdf, sum1, sum2 = maxofsum(a, b, c)

"""
print(np.max(sum1))
print("max of z is:",np.max(z))
print(np.max(MOS))
"""

plt.figure()

plt.subplot2grid((3,3),(0,0),colspan = 3)

plt.subplot2grid((3,3), (0,0))
af = sns.lineplot(x=a, y=apdf, color = "blue")
af.set(title = "apdf")
plt.subplot2grid((3,3), (0,1))
bf = sns.lineplot(x=b, y=bpdf, color = "yellow")
bf.set(title = "bpdf")
plt.subplot2grid((3,3), (0,2))
cf = sns.lineplot(x=c, y=cpdf, color = "red")
cf.set(title = "cpdf")

plt.subplot2grid((3,3), (1,0))
sum1f = sns.distplot(sum1, color = "red")
sum1f.set(title = "Sum of a and c")
plt.subplot2grid((3,3), (1,1))
sum2f = sns.distplot(sum2, color = "red")
sum2f.set(title = "Sum of b and c")

plt.subplot2grid((3,3), (2,0))
maxf = sns.lineplot(x=z, y=zpdf, color = "black")
maxf.set(title = "Max of a and b")

plt.subplot2grid((3,3), (2,1))
cf = sns.lineplot(x=c, y=cpdf, color = "red")
cf.set(title = "cpdf")

plt.subplot2grid((3,3), (2,2))
SOMf = sns.distplot(SOM, color = "red")
SOMf.set(title = "Sum of Max")
plt.subplot2grid((3,3), (1,2))
MOSf = sns.lineplot(x=MOS, y=MOSpdf, color = "blue")
MOSf.set(title = "Max of Sum")

plt.show()