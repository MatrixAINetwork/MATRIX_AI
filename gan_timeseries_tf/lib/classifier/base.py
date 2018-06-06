"""
Copyright 2018 The Matrix Authors
This file is part of the Matrix library.

The Matrix library is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The Matrix library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with the Matrix library. If not, see <http://www.gnu.org/licenses/>.
@author: Steve Deng
"""

from sklearn.svm import LinearSVC as LSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as skl_metrics
import multiprocessing
import numpy as np


def knn(X_train, y_train, X_test, y_test):
    cpu_count = multiprocessing.cpu_count()
    knn = KNeighborsClassifier(n_neighbors=1, n_jobs=max(1, cpu_count // 3))
    knn.fit(X_train, y_train)

    y_pred_train = knn.predict(X_train)
    acc_train = skl_metrics.accuracy_score(y_true=y_train, y_pred=y_pred_train)
    y_pred_test = knn.predict(X_test)
    acc_test = skl_metrics.accuracy_score(y_true=y_test, y_pred=y_pred_test)

    return acc_train, acc_test


def linearSVC(X_train, y_train, X_test, y_test):
    model = LSVC()
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    acc_train = skl_metrics.accuracy_score(y_true=y_train, y_pred=y_pred_train)
    y_pred_test = model.predict(X_test)
    acc_test = skl_metrics.accuracy_score(y_true=y_test, y_pred=y_pred_test)

    return acc_train, acc_test

def logisticRegression(X_train, y_train, X_test, y_test, Cs=[0.01, 0.05, 0.1, 0.5, 1., 5., 10., 50., 100.]):
    acc_train = []
    acc_test = []
    models = []
    for C in Cs:
        model = LogisticRegression(C=C)
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        acc_train_ = skl_metrics.accuracy_score(y_train, y_pred_train)
        acc_test_ = skl_metrics.accuracy_score(y_test, y_pred_test)
        acc_train.append(acc_train_)
        acc_test.append(acc_test_)
        models.append(model)
    best = np.argmax(acc_test)

    return acc_train[best], acc_test[best], models[best], Cs[best]