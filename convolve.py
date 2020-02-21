import numpy as np
import math
# import matplotlib
import matplotlib.pyplot as plt

# import seaborn
import seaborn as sns

# settings for seaborn plotting style
sns.set(color_codes=True,style="white")
# settings for seaborn plot sizes
sns.set(rc={'figure.figsize':(10,10)})

# import norm distribution
import scipy.stats as sci
from scipy import signal

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

def ecdf(data):
    """ Compute ECDF """
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n+1) / n
    return(x,y)
##################################### Normal distribution
"""
class gaussian_gen(sci.rv_continuous):
    '''Gaussian distribution'''
    def _pdf(self, x):
        return np.exp(-x**2 / 2.) / np.sqrt(2.0 * np.pi)
      
gaussian = gaussian_gen(name = 'gaussian')
  
x = np.arange(-4,4,0.001)
print(gaussian._pdf(x))
plt.plot(x, gaussian._pdf(x))
#ax = sns.distplot(x,
#                  bins=1000,
#                  kde=True,
#                  color='skyblue')
#ax.set(xlabel='Normal Distribution', ylabel='Frequency')

plt.show()
"""
####good pattern
mu = 2
sigma = 3
size = 5000
px = []
py = []
#x = np.random.normal(mu, sigma, size)
#Return a sample (or samples) from the “standard normal” distribution.
x = sigma * np.random.randn(size) + mu
x.sort()
print(x)
for k in x:
    px.append((1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp( - (k - mu)**2 / (2 * sigma**2)))
print(px)
sns.distplot(x,px, hist=False, color = "blue");

###numpy way
#num_bins = 5000
#counts, bin_edges = np.histogram (x, bins=num_bins, density=True)
#cdf = np.cumsum (counts)
#plt.plot (bin_edges[1:], cdf/cdf[-1])
### Scipy way
#(data, cdf) = ecdf(x)
#sns.lineplot(data, cdf);


muy = 2
sigmay = 3
sizey = 5000

#y = np.random.normal(muy, sigmay, sizey)
y = sigmay * np.random.randn(sizey) + muy
y.sort()
print(y)
for k in y:
    py.append((1 / (np.sqrt(2 * np.pi) * sigmay)) * np.exp( - (k - muy)**2 / (2 * sigmay**2)))
sns.distplot(y,py,hist=False, color = "orange");
print(py)
"""
z = np.convolve(x,y, mode='same')/sum(y)
z1 = signal.convolve(x, y, mode='same') / sum(y)
sns.distplot(z, hist=False, color = "black");
sum = sns.distplot(z1, hist=False, color = "green");
sum.set(xlabel='Blue: x, Orange: y  Green: z ', ylabel='SUM')
"""

(datax, cdfx) = ecdf(x)
sns.lineplot(datax, cdfx, color = "red")
(datay, cdfy) = ecdf(y)
sns.lineplot(datay, cdfy)


h = max(max(x), max(y))
l = max(min(x), min(y))

m =[]
for i in range(int(l),int(h)):
    if x[i] == y[i] :
        m.append(datax[i]*py[i]+px[i]*datay[i])
    else:
        idx = find_nearest(y, x[i])
        m.append(datax[i]*py[idx]+px[i]*datay[idx])
        
sns.distplot(m, hist=False, color = "green");
    
print(m)
(datam, cdfm) = ecdf(m)
sns.lineplot(datam, cdfm, color = "black")

plt.show()
