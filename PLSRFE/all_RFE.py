# coding=utf-8
# __author__=zqx

from numpy import *
from sklearn.linear_model import (LinearRegression, Ridge,
                                  Lasso, RandomizedLasso)
from sklearn.feature_selection import RFE, f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from minepy import MINE
from sklearn import preprocessing


'''
单因素方差分析后的特征剩下637种外源性物质(特征)
进行特征重要性排序
'''
def loadDataSet(filename):
    fr = open(filename)
    arrayLines = fr.readlines()[1:]
    row = len(arrayLines)
    x = mat(zeros((row, 103)))
    y = mat(zeros((row, 1)))
    index = 0
    for line in arrayLines:
        curLine = line.strip().split('\t')
        x[index,:] = curLine[0:103]
        y[index,:] = curLine[103:104]
        index += 1
    x = preprocessing.scale(x)
    y = preprocessing.scale(y)
    x = np.array(x)
    y = np.array(y)
    return x, y



X, Y= loadDataSet('RBData.txt')
print(np.shape(X))
print(np.shape(Y))
Y=Y.reshape(267)
fr = open('RBData.txt')
names1 = fr.readlines()[0:1]
for one in names1:
    names = one.strip().split('\t')
    names = names[0:103]

ranks = {}



def rank_to_dict(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order * np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x, 2), ranks)
    return dict(zip(names, ranks))



#计算线性回归系数
lr = LinearRegression(normalize=True)
lr.fit(X, Y)
ranks["Linear reg"] = rank_to_dict(np.abs(lr.coef_), names)

#F检验，根据F值的大小
f, pval = f_regression(X, Y, center=True)
ranks["f"] = rank_to_dict(f, names)

#互信息
mine = MINE()
mic_scores = []
for i in range(X.shape[1]):
    XX = X[:, i]
    mine.compute_score(X[:, i], Y)
    m = mine.mic()
    mic_scores.append(m)

ranks["MIC"] = rank_to_dict(mic_scores, names)

#L1正则
lasso = Lasso(alpha=.0012)
lasso.fit(X, Y)
ranks["Lasso"] = rank_to_dict(np.abs(lasso.coef_), names)

#L2正则
ridge = Ridge(alpha=7)
ridge.fit(X, Y)
ranks["Ridge"] = rank_to_dict(np.abs(ridge.coef_), names)

#稳定性选择
rlasso = RandomizedLasso(alpha=0.04)
rlasso.fit(X, Y)
ranks["Stability"] = rank_to_dict(np.abs(rlasso.scores_), names)

#递归特征消除
# stop the search when 5 features are left (they will get equal scores)
rfe = RFE(lr, n_features_to_select=5)
rfe.fit(X, Y)
ranks["RFE"] = rank_to_dict(list(map(float, rfe.ranking_)), names, order=-1)

#平均不纯度减少
rf = RandomForestRegressor()
rf.fit(X, Y)
ranks["MDI"] = rank_to_dict(rf.feature_importances_, names)


#平均精确率减少
r = {}
for name in names:
    r[name] = round(np.mean([ranks[method][name]
                             for method in ranks.keys()]), 2)
methods = sorted(ranks.keys())
ranks["MDA"] = r
methods.append("MDA")

print("\t%s" % "\t".join(methods))
for name in names:
    print("%s\t%s" % (name, "\t".join(map(str,
                                          [ranks[method][name] for method in methods]))))
