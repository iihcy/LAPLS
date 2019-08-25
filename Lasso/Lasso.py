# coding: utf-8

from numpy import *
from sklearn.linear_model import Lasso
import random


'''
为了获得较可靠的结果，需测试数据（测验）和训练数据（学习）
--可按照6:4的比例划分数据集,即随机分成训练集与测试集
'''
def splitDataSet(x, y):
    m =shape(x)[0]
    train_sum = int(round(m * 0.7))
    test_sum = m - train_sum
    #利用range()获得样本序列
    randomData = range(0,m)
    randomData = list(randomData)
    #根据样本序列进行分割- random.sample(A,rep)
    train_List = random.sample(randomData, train_sum)
    #获取训练集数据-train
    train_x = x[train_List,: ]
    train_y = y[train_List,: ]
    #获取测试集数据-test
    test_list = []
    for i in randomData:
        if i in train_List:
            continue
        test_list.append(i)
    test_x = x[test_list,:]
    test_y = y[test_list,:]
    return train_x, train_y, test_x, test_y

def LASSO(x, y):
    # WYHXB798 - alpha=1000, max_iter=1000, tol=1e-4; (训与测)alpha=0.00001, max_iter=1000, tol=1e-1
    # BreastData33 - alpha=0.00001, max_iter=1000, tol=1e-1
    # （训与测）alpha=0.001, max_iter=1000, tol=1e-5
    # model = Lasso(alpha=1000, max_iter=150, tol=100*e) # 训练集WYHXB
    # model = Lasso(alpha=8000, max_iter=1500, tol=e) # 测试集WYHXB
    # model = Lasso(alpha=0.0005, max_iter=15000, tol=1e-8) # 测试集TCM
    # model = Lasso(alpha=0.0001, max_iter=15000, tol=1e-10) # 训练集TCM
    # model = Lasso(alpha=0.0001, max_iter=15000, tol=1e-10) # 训练集CCrime--测试集CCrime
    # model = Lasso(alpha=0.6, max_iter=1000, tol=1e-4) # 测试集BreastData
    # model = Lasso(alpha=0.6, max_iter=1000, tol=1e-4) # 训练集BreastData
    # model = Lasso(alpha=500, max_iter=15000, tol=5e-2) # 训练集RBuild
    model = Lasso(alpha=500, max_iter=15000, tol=5e-2) # 训练集RBuild
    model.fit(x, y)
    dd = model.coef_

    ff = model.selection

    row = shape(x)[0]
    mean_y = mean(y, 0)
    y_mean = tile(mean_y, (row, 1))
    # 对实验数据进行预测
    # model.predict()
    y_predicts = model.predict(x)
    y_predict = mat(y_predicts).T
    RF_SSE = sum(power((y_predict - y), 2), 0)
    RF_SST = sum(sum(power((y - y_mean), 2), 0))
    RF_SSR = sum(sum(power((y_predict - y_mean), 2), 0))
    RF_RMSE = sqrt(RF_SSE / row)  # 均方根误差
    R_squared = RF_SSR / RF_SST
    return R_squared, RF_RMSE, y_predict


#数据读取-单因变量与多因变量(WYHXB-407)
def loadDataSet(filename):
    fr = open(filename)
    arrayLines = fr.readlines()
    row = len(arrayLines)
    x = mat(zeros((row, 103)))
    y = mat(zeros((row, 1)))
    index = 0
    for line in arrayLines:
        curLine = line.strip().split('\t')
        x[index, :] = curLine[0:103]
        y[index, :] = curLine[-1]
        index += 1
    return x, y


if __name__ == '__main__':
    x, y = loadDataSet('RBuild.txt')
    # 训练集与测试集 - 0.6与0.4
    train_x, train_y, test_x, test_y = splitDataSet(x, y)

    R_squared, RMSE, y_predict = LASSO(test_x, test_y)
    y_predict = mat(y_predict).T
    print('========================')
    print('R_squared:', R_squared)
    print('RMSE:', RMSE)
    print('========================')
    # print('y_predict:', y_predict)
    print(shape(x), shape(y))
    # print(shape(train_x), shape(train_y))