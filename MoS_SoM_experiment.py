import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import random
import scipy
import seaborn as sns
sns.set(style="white", palette="muted", color_codes=True)
sns.set(font_scale=0.5)
# settings for seaborn plot sizes
sns.set(rc={'figure.figsize':(12,12)})


def NORM(mu, sigma, size):
    x = sigma * np.random.randn(size) + mu
    x = np.around(x, decimals=2)
    cx = scipy.stats.norm.cdf(x, loc=mu, scale=sigma)
    px = scipy.stats.norm.pdf(x, loc=mu, scale=sigma)
    px = px/sum(px)################################################################
    f = pd.DataFrame({'Data': x, 'PDF': px, 'CDF': cx})
    print(sum(px))
    return f
def SUM(f1, f2):
    P = []
    M = []
    for i in range(f1['Data'].size):
        for j in range(f2['Data'].size):
            value = f1['Data'][i]+f2['Data'][j]
            if (f1['Data'][i]+f2['Data'][j]) not in M:
                M.append(f1['Data'][i]+f2['Data'][j])
                P.append(f1['PDF'][i]*f2['PDF'][j])
            else:
                P[M.index(f1['Data'][i]+f2['Data'][j])] = P[M.index(f1['Data'][i]+f2['Data'][j])] + f1['PDF'][i]*f2['PDF'][j]
    f = pd.DataFrame({'Data': M, 'PDF': P})
    return f
def MAX(f1,f2):
    P = []
    M = []
    for i in range(f1['Data'].size):
        P_temp = 0
        for j in range(f2['Data'].size):
            if (f1['Data'][i] >= f2['Data'][j]):
                P_temp = P_temp + f1['PDF'][i]*f2['PDF'][j]
        if f1['Data'][i] not in M:
            M.append(f1['Data'][i])
            P.append(P_temp)
        else:
            P[M.index(f1['Data'][i])] = P[M.index(f1['Data'][i])] + P_temp

    for i in range(f2['Data'].size):
        P_temp = 0
        for j in range(f1['Data'].size):
            if (f2['Data'][i] > f1['Data'][j]):
                P_temp = P_temp + f2['PDF'][i]*f1['PDF'][j]
        if f2['Data'][i] not in M:
            M.append(f2['Data'][i])
            P.append(P_temp)
        else:
            P[M.index(f2['Data'][i])] = P[M.index(f2['Data'][i])] + P_temp
    f = pd.DataFrame({'Data': M, 'PDF': P})
    return f

def MAX_of_SUM(f1,f2,fc):
    fs1 = SUM(f1,fc)
    fs2 = SUM(f2,fc)
    fms = MAX(fs1,fs2)
    return fms

def SUM_of_MAX(f1,f2,fc):
    fm1 = MAX(f1,fc)
    fm2 = MAX(f2,fc)
    fsm = SUM(fm1,fm2)
    return fsm


def Result_plot(f1, f2, fc,fsm, fms):
    plt.figure()
    plt.xticks(fontsize=3)
    plt.yticks(fontsize=3)
    plt.subplot2grid((3,3), (0, 0))
    #######Start to plot
    F1 = plt.subplot2grid((3, 3), (0, 0))
    F1.title.set_text('Mean = 8, Std = 1, Size = 100')
    F1.set_xticks([])
    F1.set_yticks([])
    f1_p = f1.sort_values(by='PDF')
    #sns.distplot(f1_p['Data'], f1_p['PDF'], color="red", label='1st input')
    sns.scatterplot(f1_p['Data'], f1_p['PDF'], color="red", label='1st input')

    F2 = plt.subplot2grid((3, 3), (0, 1))
    F2.title.set_text('Mean = 8, Std = 1, Size = 100')
    F2.set_xticks([])
    F2.set_yticks([])
    f2_p = f2.sort_values(by='PDF')
    sns.scatterplot(f2_p['Data'], f2_p['PDF'], color="green", label='2nd input')

    FC = plt.subplot2grid((3, 3), (0, 2))
    FC.title.set_text('Mean = 5, Std = 1, Size = 70')
    FC.set_xticks([])
    FC.set_yticks([])
    fc_p = fc.sort_values(by='PDF')
    sns.scatterplot(fc_p['Data'], fc_p['PDF'], color="blue", label='3rd input')
#"""
    SMP = plt.subplot2grid((3, 3), (1, 0),colspan=3)
    #SMP.title.set_text('SUM_of_MAX')
    fsm_p = fsm[fsm.PDF != 0]
    fsm_p = fsm_p.sort_values(by='PDF')
    sns.scatterplot(fsm_p['Data'], fsm_p['PDF'], color="tan", label='SUM_of_MAX')

    MSP = plt.subplot2grid((3, 3), (2, 0), colspan=3)
    #MSP.title.set_text('SUM_of_MAX')
    fms_p = fms[fms.PDF != 0]
    fms_p = fms_p.sort_values(by='PDF')
    sns.scatterplot(fms_p['Data'], fms_p['PDF'], color="teal", label='MAX_of_SUM')
#"""


mu1 = 8
sigma1 = 1
size1 = 100
f1 = NORM(mu1, sigma1, size1)

mu2 = 8
sigma2 = 1
size2 = 100
f2 = NORM(mu2, sigma2, size2)

mu3 = 5
sigma3 = 1
size3 = 70
fc = NORM(mu3, sigma3, size3)

"""
fsm = SUM_of_MAX(f1,f2,fc)
#print(fsm)

#print("finish SUM of MAX part")

fms = MAX_of_SUM(f1,f2,fc)
#print(fms)

Result_plot(f1,f2,fc,fsm,fms)
"""

fsum = SUM(f1,f2)

plt.figure()
plt.xticks(fontsize=3)
plt.yticks(fontsize=3)
plt.subplot2grid((2,2), (0, 0))
#######Start to plot
F1 = plt.subplot2grid((2, 2), (0, 0))
F1.title.set_text('Mean = 8, Std = 1, Size = 100')
F1.set_xticks([])
F1.set_yticks([])
f1_p = f1.sort_values(by='PDF')
#sns.distplot(f1_p['Data'], f1_p['PDF'], color="red", label='1st input')
sns.scatterplot(f1_p['Data'], f1_p['PDF'], color="red", label='1st input')

F2 = plt.subplot2grid((2, 2), (0, 1))
F2.title.set_text('Mean = 8, Std = 1, Size = 100')
F2.set_xticks([])
F2.set_yticks([])
f2_p = f2.sort_values(by='PDF')
sns.scatterplot(f2_p['Data'], f2_p['PDF'], color="green", label='2nd input')

FS = plt.subplot2grid((2, 2), (1, 0), colspan = 2)
FS.set_xticks([])
FS.set_yticks([])
fc_p = fsum.sort_values(by='PDF')
sns.scatterplot(fc_p['Data'], fc_p['PDF'], color="blue", label='SUM')
plt.show()


