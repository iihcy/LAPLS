#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
from math import exp

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from standard_linear_regression import load_data,load_data00, standarize, get_corrcoef

#单因变量与多因变量
def loadDataSet(fileName):
    #获取样本特征的总数
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            #将数据添加到lineArr List中
            lineArr.append(float(curLine[i]))
            #将测试数据的输入数据部分存储到dataMat
        dataMat.append(lineArr)
        # 将每一行的类别存储到labelMat
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

#多因变量与多因变量
def loadDataSet00(fileName):
    #获取样本特征的总数
    numFeat = len(open(fileName).readline().split('\t')) - 2
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            #将数据添加到lineArr List中
            lineArr.append(float(curLine[i]))
            #将测试数据的输入数据部分存储到dataMat
        dataMat.append(lineArr)
        # 将每一行的类别存储到labelMat
        labelMat.append(float(curLine[-2:-1]))
    return dataMat, labelMat



def lasso_regression(X, y, lambd=0.2, threshold=0.1):
    '''
        通过坐标下降(coordinate descent)法获取LASSO回归系数
    '''
    #计算残差平方和
    rss = lambda X, y, w: (y - X*w).T*(y - X*w)
    # 初始化回归系数w.
    m, n = X.shape
    w = np.matrix(np.zeros((n, 1)))
    r = rss(X, y, w)
    #使用坐标下降法优化回归系数w
    niter = itertools.count(1)
    #lambd = lambd
    for it in niter:
        print(u'lasso_regression：', it)
        for k in range(n):
            # 计算常量值z_k和p_k
            AA=X[:, k].T*X[:, k]
            z_k = (X[:, k].T*X[:, k])[0, 0]
            p_k = 0
            for i in range(m):
                a=X[i, k]
                b=y[i, 0]
                #cc=X[i, j]
                #ccc=X[i, j]*w[j, 0]
                #c=[X[i, j]*w[j, 0] for j in range(n) if j != k]
                #d=sum([X[i, j]*w[j, 0] for j in range(n) if j != k])
                p_k += X[i, k]*(y[i, 0] - sum([X[i, j]*w[j, 0] for j in range(n) if j != k]) )
                #ff=X[i, j]
                #dd=w[j, 0]
                #hh=X[i, j]*w[j, 0]

            la = -lambd/2
            if p_k < -lambd/2:
                w_k = (p_k + lambd/2)/z_k
            elif p_k > lambd/2:
                w_k = (p_k - lambd/2)/z_k
            else:
                w_k = 0
            w[k, 0] = w_k
        r_prime = rss(X, y, w)
        delta = abs(r_prime - r)[0, 0]
        r = r_prime
        #print('Iteration: {}, delta = {}'.format(it, delta))
        if delta < threshold:
            break
    return w

def lasso_traj(X, y, ntest=30):
    ''' 获取回归系数轨迹矩阵
    '''
    _, n = X.shape
    ws = np.zeros((ntest, n))
    for i in range(ntest):
        print(u'lasso_traj【30次】ntest:', i)
        w = lasso_regression(X, y, lambd=exp(i-10))
        ws[i, :] = w.T
        #print('lambda = e^({}), w = {}'.format(i-10, w.T[0, :]))
    return ws



if '__main__' == __name__:
    # 对于单因变量
    X, y = load_data('NYHXB.txt')
    #对于多因变量
    X, y = standarize(X), standarize(y)
    ntest = 30
    #绘制轨迹
    ws = lasso_traj(X, y, ntest)
    #ws = lasso_traj(XX, yy, ntest)
    print (u'回归系数轨迹ws:', ws, np.shape(ws))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    lambdas = [i-0 for i in range(ntest)]
    plt.title(u'NYHXB')
    plt.xlabel('Iteration')
    plt.ylabel('w')
    ax.plot(lambdas, ws)
    plt.show()
    file = pd.DataFrame(ws)
    file.to_csv(u"NYHXB迭代回归系数.csv")