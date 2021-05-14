from sklearn.metrics import balanced_accuracy_score
import math
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import pairwise_distances


def trainTestEvaluation(X_train, y_train, X_test, y_test, clf):
    clf.fit(X_train, y_train)
    y_test_predict = clf.predict(X_test)
    error = 1.0 - balanced_accuracy_score(y_true=y_test, y_pred=y_test_predict)
    return error


def kFoldCrossValidation(X, y, clf, skf):
    no_folds = skf.n_splits

    error = 0.0
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx, :], X[test_idx, :]
        y_train, y_test = y[train_idx], y[test_idx]
        clf.fit(X_train, y_train)
        y_test_pred = clf.predict(X_test)
        error += 1.0 - balanced_accuracy_score(y_true=y_test, y_pred=y_test_pred)
    error = error / no_folds

    return error


def LOOCV_NN(X: np.array, y: np.array, k):
    nbrs = NearestNeighbors(n_neighbors=k + 1, metric="precomputed")
    distance = pairwise_distances(X)
    nbrs.fit(distance)
    _, indices = nbrs.kneighbors(distance)
    nn_indices = indices[:, 1:(k + 1)]
    y_pred = np.array([])
    for nn_idx in nn_indices:
        y_neighbors = y[nn_idx]
        assert len(y_neighbors) == k
        labels, counts = np.unique(y_neighbors, return_counts=True)
        label = labels[np.argmax(counts)]
        y_pred = np.append(y_pred, label)
    error = 1.0 - balanced_accuracy_score(y_true=y, y_pred=y_pred)
    return error


def population_stat(pop: np.array):
    pop_mean = np.mean(pop, axis=0)
    diff_mean = pop - pop_mean
    diff_mean = diff_mean ** 2
    diverse = np.mean([np.sqrt(np.sum(diff_mean, axis=1))])
    return diverse


def sigmoid(value):
    return 1.0 / (1 + math.exp(-value))


# if __name__ == '__main__':
#     import time
#
#     X = np.random.rand(3000, 1000)
#     y = np.random.randint(low=1, high=5, size=3000)
#
#     start = time.time()
#     print(LOOCV_NN(X, y, k=1))
#     exe_time = time.time() - start
#     print(exe_time)
#     print()
#
#     start = time.time()
#     print(LOOCV_NN(X, y, k=3))
#     exe_time = time.time() - start
#     print(exe_time)
#     print()
#
#     start = time.time()
#     print(LOOCV_NN(X, y, k=5))
#     exe_time = time.time() - start
#     print(exe_time)
