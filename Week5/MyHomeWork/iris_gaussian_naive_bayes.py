#!/usr/bin/env python
# coding: utf-8

# 使用sklearn的高斯贝叶斯接口实现鸢尾花分类

from sklearn.naive_bayes import GaussianNB
from sklearn import datasets

iris = datasets.load_iris()

gnb = GaussianNB()

y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d"
      % (iris.data.shape[0], (iris.target != y_pred).sum()))
