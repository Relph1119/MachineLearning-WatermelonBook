#!/usr/bin/env python
# encoding: utf-8
"""
@author: HuRuiFeng
@file: homework_2.py
@time: 2019/3/29 22:03
@desc:
"""

"""导入所需要的软件包"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

"""
# @代码功能简介：从硬盘上读取已下载好的数据，并进行简单处理
# @知识点定位：数据预处理
"""
df_train = pd.read_csv('./new_data/train_set.csv')
df_test = pd.read_csv('./new_data/test_set.csv')
df_train.drop(columns=['article', 'id'], inplace=True)
df_test.drop(columns=['article'], inplace=True)

"""
# @代码功能简介，将数据集中的字符文本转换成数字向量，以便计算机能够进行处理（一段文字 ----> 一个向量）
# @知识点定位：特征工程
"""
vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.9, max_features=100000)
vectorizer.fit(df_train['word_seg'])
x_train = vectorizer.transform(df_train['word_seg'])
x_test = vectorizer.transform(df_test['word_seg'])
y_train = df_train['class'] - 1

"""
# @代码功能简介：训练一个分类器
# @知识点定位：传统监督学习算法之线性逻辑回归模型
"""
lg = LogisticRegression(C=4, dual=True, n_jobs=-1)
lg.fit(x_train, y_train)

"""根据上面训练好的分类器对测试集的每个样本进行预测"""
y_test = lg.predict(x_test)

"""将测试集的预测结果保存至本地"""
df_test['class'] = y_test.tolist()
df_test['class'] = df_test['class'] + 1
df_result = df_test.loc[:, ['id', 'class']]
df_result.to_csv('./result.csv', index=False)