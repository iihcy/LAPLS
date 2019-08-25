#coding:utf-8

import numpy as np
import pandas as pd
import re
'''
    7种算法交集后的特征与数据--30种的数据集
'''


allData = pd.read_csv("WYXWZ.csv")
##获取所有的列名
allcolName = allData.columns.values.tolist()
m = allData.shape[1]
mm = allData.shape[0]
print(m, mm)
selectData = pd.read_csv("mat.csv")
selectData = np.mat(selectData)
n = selectData.shape[0]
print(n)
allColName = np.array(allcolName)
selectData = np.mat(selectData)
AllselectDatas = []
for i in range(m):
    print('Iter:', i)
    allNameI = allcolName[i]
    for j in range(n):
        f_s = selectData[j, :]
        f_str = str(f_s)
        s_name = re.sub('[[\\]]', '', f_str)
        sJ_Name = re.sub('\'', '', s_name)
        if allNameI == sJ_Name:
            allData = np.mat(allData)
            ff= allData[:, i]
            AllselectDatas.append(ff)

AllselectDatas = np.array(AllselectDatas)
AllselectDatas = np.mat(AllselectDatas).T
# AllselectDatas = np.array(AllselectDatas)
# selectData = np.array(selectData)
# selectData = np.mat(selectData).T
# selectData = np.array(selectData)
# AselData = np.vstack((selectData, AllselectDatas))

file = pd.DataFrame(AllselectDatas)
file.to_csv(u'TrainWYHXB数据.csv')