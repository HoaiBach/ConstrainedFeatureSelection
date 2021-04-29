from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold as SKF
from sklearn.model_selection import KFold
import numpy as np
import math


def trainTestEvaluation(X_train, y_train, X_test, y_test, clf):
    clf.fit(X_train, y_train)
    y_test_predict = clf.predict(X_test)
    error = 1.0 - balanced_accuracy_score(y_true=y_test, y_pred=y_test_predict)
    return error


def kFoldCrossValidation(X, y, clf, k=5):
    labels, counts = np.unique(y, return_counts=True)
    label_min = np.min(counts)
    if label_min < k:
        skf = KFold(n_splits=k, shuffle=True, random_state=1617)
    else:
        skf = SKF(n_splits=k, shuffle=True, random_state=1617)

    error = 0.0
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx, :], X[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]
        error += trainTestEvaluation(X_train, y_train, X_test, y_test, clf=clf)
    error = error / k
    return error


def sigmoid(value):
    return 1.0 / (1 + math.exp(-value))
