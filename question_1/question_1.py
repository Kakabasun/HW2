# -*- coding: utf-8 -*-
"""

@author: Ruifeng Song
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
import pandas as pd

# Part A
m01 = np.array([3, 0])
m02 = np.array([0, 3])
m1 = np.array([2, 2])
C01 =  np.array([[2, 0], [0, 1]])
C02 =  np.array([[1, 0], [0, 2]])
C1 = np.array([[1, 0], [0, 1]])

# testdata = np.zeros(10000, dtype='4float32')  # 10,0000 test data points; 1d array of tuples
# classes = np.random.rand(10000) >= .7


# for i in range(10000):
#     if(classes[i]):
#         testdata[i] = np.random.multivariate_normal(m1, C1)
#     else:
#         testdata[i] = np.random.multivariate_normal(m0, C0)

    
problem1testdata = np.load(r'C:\Users\Allen Song\Desktop\Courses\Machine Learning\hw2\problem1testdata.npy')
problem1classes = np.load(r'C:\Users\Allen Song\Desktop\Courses\Machine Learning\hw2\problem1classes.npy')

pL0 = 0.5*multivariate_normal.pdf(problem1testdata, m01, C01) + 0.5*multivariate_normal.pdf(problem1testdata, m02, C02)
pL1 = multivariate_normal.pdf(problem1testdata, m1, C1)
likelihoodratio = pL1/pL0

maxgamma = np.amax(likelihoodratio[np.logical_not(problem1classes)])

testgammas = np.arange(0, maxgamma, maxgamma/10000)
resultdata = pd.DataFrame(columns = ['Gamma', 'True 1', 'False 1', 'False 0', 'True 0']);
resultdata['Gamma'] = testgammas

for i in range(10000):

    classifierresults = likelihoodratio > testgammas[i]
    resultdata.loc[resultdata.index[i], 'True 1'] = np.sum(np.logical_and(classifierresults, problem1classes))/10000
    resultdata.loc[resultdata.index[i], 'True 0'] = np.sum(np.logical_and(np.logical_not(classifierresults), np.logical_not(problem1classes)))/10000
    resultdata.loc[resultdata.index[i], 'False 1'] = np.sum(np.logical_and(classifierresults, np.logical_not(problem1classes)))/10000
    resultdata.loc[resultdata.index[i], 'False 0'] = np.sum(np.logical_and(np.logical_not(classifierresults), problem1classes))/10000
    
ax = resultdata.plot(x='False 1', y='True 1')
ax.set_xlabel('False Positives')
ax.set_ylabel('True Positives')
ax.set_title('ROC Curve')

resultdata['p_error'] = resultdata['False 1']*.7+resultdata['False 0']*.3

optimumgamma_idx = resultdata['p_error'].astype(float).idxmin()

optimumgamma = 1.857
classifierresults = likelihoodratio > optimumgamma
optimumhit = np.sum(np.logical_and(classifierresults, problem1classes))/10000
optimumfalsepoz = np.sum(np.logical_and(classifierresults, np.logical_not(problem1classes)))/10000
optimumfalseneg = np.sum(np.logical_and(np.logical_not(classifierresults), problem1classes))/10000
optimum_p_error = optimumfalsepoz*.65 + optimumfalseneg*.35

ax.plot(resultdata.loc[resultdata.index[optimumgamma_idx], 'False 1'], resultdata.loc[resultdata.index[optimumgamma_idx], 'True 1'], 'rx')
ax.plot(optimumfalsepoz, optimumhit, 'bx')
ax.annotate("Empirical Optimum: (%f, %f)" % (resultdata.loc[resultdata.index[optimumgamma_idx], 'False 1'], resultdata.loc[resultdata.index[optimumgamma_idx], 'True 1']), (resultdata.loc[resultdata.index[optimumgamma_idx], 'False 1'], resultdata.loc[resultdata.index[optimumgamma_idx], 'True 1']))
ax.annotate("Theoretical Optimum: (%f, %f)" % (optimumfalsepoz, optimumhit), (optimumfalsepoz, optimumhit), textcoords='offset pixels')

# Part B

C01_naive = np.diag(np.diagonal(C01))
C02_naive = np.diag(np.diagonal(C02))
C1_naive = np.diag(np.diagonal(C1))

pL0 = 0.5*multivariate_normal.pdf(problem1testdata, m01, C01) + 0.5*multivariate_normal.pdf(problem1testdata, m02, C02)
pL1 = multivariate_normal.pdf(problem1testdata, m1, C1)
pL0_naive = multivariate_normal.pdf(problem1testdata, m0, C0_naive)
pL1_naive = multivariate_normal.pdf(problem1testdata, m1, C1_naive)
likelihoodratio_naive = pL1_naive/pL0_naive

resultdata_naive = pd.DataFrame(columns = ['Gamma', 'True 1', 'False 1', 'False 0', 'True 0']);
resultdata_naive['Gamma'] = testgammas

for i in range(10000):
    classifierresults = likelihoodratio_naive > testgammas[i]
    resultdata_naive.loc[resultdata_naive.index[i], 'True 1'] = np.sum(np.logical_and(classifierresults, problem1classes))/10000
    resultdata_naive.loc[resultdata_naive.index[i], 'True 0'] = np.sum(np.logical_and(np.logical_not(classifierresults), np.logical_not(problem1classes)))/10000
    resultdata_naive.loc[resultdata_naive.index[i], 'False 1'] = np.sum(np.logical_and(classifierresults, np.logical_not(problem1classes)))/10000
    resultdata_naive.loc[resultdata_naive.index[i], 'False 0'] = np.sum(np.logical_and(np.logical_not(classifierresults), problem1classes))/10000

ax = resultdata_naive.plot(x='False 1', y='True 1')
resultdata.plot(ax=ax, x='False 1', y='True 1')
ax.set_xlabel('False Positives')
ax.set_ylabel('True Positives')
ax.set_title('Naive ROC Curve')

resultdata_naive['p_error'] = resultdata_naive['False 1']*.65+resultdata_naive['False 0']*.35

classifierresults_naive = likelihoodratio_naive > optimumgamma
optimumhit_naive = np.sum(np.logical_and(classifierresults_naive, problem1classes))/10000
optimumfalsepoz_naive = np.sum(np.logical_and(classifierresults_naive, np.logical_not(problem1classes)))/10000
optimumfalseneg_naive = np.sum(np.logical_and(np.logical_not(classifierresults_naive), problem1classes))/10000
optimum_p_error_naive = optimumfalsepoz_naive*.65 + optimumfalseneg_naive*.35

optimumgamma_idx = resultdata_naive['p_error'].astype(float).idxmin()


ax.plot(resultdata_naive.loc[resultdata_naive.index[optimumgamma_idx], 'False 1'], resultdata_naive.loc[resultdata_naive.index[optimumgamma_idx], 'True 1'], 'rx')
ax.plot(optimumfalsepoz_naive, optimumhit_naive, 'bx')
ax.annotate("Empirical Naive Optimum: (%f, %f)" % (resultdata_naive.loc[resultdata_naive.index[optimumgamma_idx], 'False 1'], resultdata_naive.loc[resultdata_naive.index[optimumgamma_idx], 'True 1']), (resultdata_naive.loc[resultdata_naive.index[optimumgamma_idx], 'False 1'], resultdata_naive.loc[resultdata.index[optimumgamma_idx], 'True 1']))
ax.annotate("Theoretical Optimum: (%f, %f)" % (optimumfalsepoz_naive, optimumhit_naive), (optimumfalsepoz_naive, optimumhit_naive), textcoords='offset pixels')

ax.legend(["Naive", "Non-Naive"])