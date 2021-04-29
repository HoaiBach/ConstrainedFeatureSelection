from scipy import io
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
import numpy as np
import random
from skfeature.utility.entropy_estimators import *


if __name__ == '__main__':
    import sys
    datasets = ['Arrhythmia', 'GuesterPhase', 'Australian', 'BASEHOCK', 'COIL20', 'Chess', 'Connect4', 'Dermatology',
                'GasSensor', 'German', 'Glass', 'ImageSegmentation', 'Hillvalley', 'Movementlibras', 'Ionosphere',
                'Isolet', 'LSVT', 'Leaf', 'LedDisplay', 'Lymph', 'Madelon', 'Mice', 'Parkinson', 'ORL',
                'MultipleFeatures', 'Musk1', 'PCMAC', 'Plant', 'RELATHE', 'Semeion', 'Sonar', 'Spect', 'WallRobot',
                'USPS', 'Vehicle', 'WBCD', 'Wine', 'Yale', 'Zoo', 'QsarOralToxicity', 'QsarAndrogenReceptor',
                'Gametes', 'Bioresponse', 'Christine', 'Gisette']
    strategy = 'TrainTest'
    datasets = ['11Tumor', '9Tumor', 'ALLAML', 'AR10P', 'Brain1', 'Brain2', 'CLL-SUB-111', 'CNS', 'Carcinom', 'Colon',
                'DLBCL', 'GLI-85', 'GLIOMA', 'Leukemia', 'Leukemia1', 'Leukemia2', 'Ovarian', 'PIE10P', 'Prostate-GE',
                'Prostate', 'SMK-CAN-187', 'SRBCT', 'TOX-171', 'arcene', 'lymphoma', 'nci9', 'orlraws10P', 'pixraw10P',
                'warpAR10P', 'warpPIE10P']
    strategy = '10Fold'
    in_dir = '/home/nguyenhoai2/Grid/data/FSMatlab/'
    out_dir = '/home/nguyenhoai2/Grid/data/PreSplit/'
    for dataset in datasets:

    # dataset = sys.argv[1]
    # strategy = sys.argv[2]
    # in_dir = sys.argv[3]
    # out_dir = sys.argv[4]
        to_write = ''
        np.random.seed(1617)
        random.seed(1617)

        # read data
        mat = io.loadmat(in_dir + dataset + '.mat')
        X = mat['X']
        X = X.astype(float)
        y = mat['Y']
        y = y[:, 0]

        if strategy == 'TrainTest':
            indices = np.arange(len(X))
            X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(X, y, indices,
                                                                                             test_size=0.3, stratify=y,
                                                                                             random_state=1617,
                                                                                             shuffle=True)
            no_instance, no_feature = X_train.shape

            to_write += 'Train indices: %s\n' % (', '.join([str(ele) for ele in train_indices]))
            to_write += 'Test indices: %s\n' % (', '.join([str(ele) for ele in test_indices]))

            # # check whether should use discrete or estimation approach
            # check_integer = []
            # for f_i in np.arange(X_train.shape[1]):
            #     X_i = X_train[:, f_i]
            #     check_integer.append(np.array([ele.is_integer() for ele in X_i]).all())
            # if np.array(check_integer).all():
            #     discrete = True
            # else:
            #     discrete = False
            #
            # # calculate the relevance between feature and class label
            # rel = []
            # for f_i in np.arange(X_train.shape[1]):
            #     X_i = X_train[:, f_i]
            #     if discrete:
            #         rel_i = midd(X_i, y_train)
            #     else:
            #         rel_i = mi(X_i[:, np.newaxis], y_train[:, np.newaxis])
            #     rel.append(rel_i)
            #
            # rel = np.array(rel)
            # rel = (rel - np.min(rel)) / (np.max(rel) - np.min(rel))
            #
            # to_write += '\nRelevance:\n'
            # to_write += '%s \n' % (', '.join([str(ele) for ele in rel]))
            #
            # # calculate redundancy
            # red = np.zeros((no_feature, no_feature), dtype=float)
            # red_cond = np.zeros((no_feature, no_feature), dtype=float)
            # for f_i in np.arange(no_feature):
            #     for f_j in np.arange(no_feature):
            #         if f_j >= f_i:
            #             X_i = X_train[:, f_i]
            #             X_j = X_train[:, f_j]
            #             if discrete:
            #                 red[f_i, f_j] = red[f_j, f_i] = midd(X_i, X_j)
            #                 red_cond[f_i][f_j] = red_cond[f_j][f_i] = cmidd(X_i, X_j, y_train)
            #             else:
            #                 red[f_i, f_j] = red[f_j, f_i] = mi(X_i[:, np.newaxis], X_j[:, np.newaxis])
            #                 red_cond[f_i][f_j] = red_cond[f_j][f_i] = cmi(X_i[:, np.newaxis], X_j[:, np.newaxis],
            #                                                               y_train[:, np.newaxis])
            # red = np.array(red)
            # red = (red - np.min(red)) / (np.max(red) - np.min(red))
            # red_cond = np.array(red_cond)
            # red_cond = (red_cond - np.min(red_cond)) / (np.max(red_cond) - np.min(red_cond))
            #
            # to_write += '\nRedundancy:\n'
            # for row in red:
            #     to_write += '%s \n' % (', '.join([str(ele) for ele in row]))
            #
            # to_write += '\nConditional Redundancy:\n'
            # for row in red_cond:
            #     to_write += '%s \n' % (', '.join([str(ele) for ele in row]))

        elif strategy == '10Fold':
            _, counts = np.unique(y, return_counts=True)
            if np.min(counts) < 10:
                skf = KFold(n_splits=10, shuffle=True, random_state=1617)
            else:
                skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1617)
            fold_idx = 1
            for train_indices, test_indices in skf.split(X, y):
                X_train, y_train = X[train_indices], y[train_indices]
                no_instance, no_feature = X_train.shape

                to_write += '====Fold %d====\n' % fold_idx
                fold_idx += 1
                to_write += 'Train indices: %s\n' % (', '.join([str(ele) for ele in train_indices]))
                to_write += 'Test indices: %s\n' % (', '.join([str(ele) for ele in test_indices]))
            #
            #     # check whether should use discrete or estimation approach
            #     check_integer = []
            #     for f_i in np.arange(X_train.shape[1]):
            #         X_i = X_train[:, f_i]
            #         check_integer.append(np.array([ele.is_integer() for ele in X_i]).all())
            #     if np.array(check_integer).all():
            #         discrete = True
            #     else:
            #         discrete = False
            #
            #     # calculate the relevance between feature and class label
            #     rel = []
            #     for f_i in np.arange(X_train.shape[1]):
            #         X_i = X_train[:, f_i]
            #         if discrete:
            #             rel_i = midd(X_i, y_train)
            #         else:
            #             rel_i = mi(X_i[:, np.newaxis], y_train[:, np.newaxis])
            #         rel.append(rel_i)
            #     rel = np.array(rel)
            #     rel = (rel - np.min(rel)) / (np.max(rel) - np.min(rel))
            #     to_write += '\nRelevance:\n'
            #     to_write += '%s \n' % (', '.join([str(ele) for ele in rel]))
            #
            #     # calculate redundancy
            #     red = np.zeros((no_feature, no_feature), dtype=float)
            #     red_cond = np.zeros((no_feature, no_feature), dtype=float)
            #     for f_i in np.arange(no_feature):
            #         for f_j in np.arange(no_feature):
            #             if f_j >= f_i:
            #                 X_i = X_train[:, f_i]
            #                 X_j = X_train[:, f_j]
            #                 if discrete:
            #                     red[f_i, f_j] = red[f_j, f_i] = midd(X_i, X_j)
            #                     red_cond[f_i][f_j] = red_cond[f_j][f_i] = cmidd(X_i, X_j, y_train)
            #                 else:
            #                     red[f_i, f_j] = red[f_j, f_i] = mi(X_i[:, np.newaxis], X_j[:, np.newaxis])
            #                     red_cond[f_i][f_j] = red_cond[f_j][f_i] = cmi(X_i[:, np.newaxis], X_j[:, np.newaxis],
            #                                                                   y_train[:, np.newaxis])
            #
            #     red = np.array(red)
            #     red = (red - np.min(red)) / (np.max(red) - np.min(red))
            #     red_cond = np.array(red_cond)
            #     red_cond = (red_cond - np.min(red_cond)) / (np.max(red_cond) - np.min(red_cond))
            #
            #     to_write += '\nRedundancy:\n'
            #     for row in red:
            #         to_write += '%s \n' % (', '.join([str(ele) for ele in row]))
            #
            #     to_write += '\nConditional Redundancy:\n'
            #     for row in red_cond:
            #         to_write += '%s \n' % (', '.join([str(ele) for ele in row]))

        f = open(out_dir + dataset + '.txt', 'w')
        f.write(to_write)
        f.close()
