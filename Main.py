import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import balanced_accuracy_score
import time
import CSO
from Utility import DataHandle
from Problem import FeatureSelection
import WorldPara
from Utility import Helpers


if __name__ == '__main__':
    import sys

    dataset = sys.argv[1]
    run = int(sys.argv[2])
    in_dir = sys.argv[3] # where is the FSmatlab
    out_dir = sys.argv[4] # where to write run.txt
    parallel = sys.argv[5] == 'parallel' # parallel or not
    if sys.argv[6].startswith('constrained'):
        # not_constrained/constrained-single-fit/constrained-single-err/constrained-hybrid
        splits = sys.argv[6].split('-')
        WorldPara.CONSTRAIN_MODE = splits[1]
        if WorldPara.CONSTRAIN_MODE == 'single':
            WorldPara.CONSTRAIN_TYPE = splits[2]
        elif WorldPara.CONSTRAIN_MODE == 'hybrid':
            WorldPara.CONSTRAIN_TYPE = 'err'
        else:
            raise Exception('%s mode is not implemented!' % WorldPara.CONSTRAIN_MODE)

    seed = 1617 * run
    np.random.seed(seed)
    random.seed(seed)

    folds = DataHandle.load_data(in_dir, dataset)
    fold_idx = 1
    to_print = 'Parallel: %s\n' % str(parallel)
    to_print += 'Constrain mode: %s\n' % str(WorldPara.CONSTRAIN_MODE)
    if WorldPara.CONSTRAIN_MODE == 'single':
        to_print += 'Constrain type: %s\n' % str(WorldPara.CONSTRAIN_TYPE)

    full_test_accs = []
    sel_accs = []
    f_ratios = []
    running_times = []
    clf = KNN(1)

    for fold in folds:
        to_print += '====Fold %d====\n' % fold_idx
        fold_idx += 1
        X_train, X_test, y_train, y_test = fold
        no_train, no_fea = X_train.shape

        # normalise data
        x_max = np.max(X_train, axis=0)
        x_min = np.min(X_train, axis=0)
        X_train = (X_train - x_min) / (x_max - x_min)
        X_test = (X_test - x_min) / (x_max - x_min)
        X_train = np.nan_to_num(X_train)
        X_test = np.nan_to_num(X_test)

        # get info for all features
        to_print += '%d training instances, %d features\n' % X_train.shape
        clf.fit(X_train, y_train)
        y_test_pred = clf.predict(X_test)
        acc = balanced_accuracy_score(y_true=y_test, y_pred=y_test_pred)
        full_test_accs.append(acc)
        to_print += '%s: Full testing accuracy %f\n' % (clf, acc)

        start = time.time()

        prob = FeatureSelection(X=X_train, y=y_train, classifier=clf,
                                init_style='Bing', fratio_weight=0.02)

        # cross_train_err = Helpers.kFoldCrossValidation(X_train, y_train, prob.clf, prob.skf)
        # cross_train_err = Helpers.LOOCV_1NN(X_train, y_train)
        # if WorldPara.ERR_CONSTRAIN:
        #     cond_constrain = cross_train_err
        # else:
        #     cond_constrain = (1.0-prob.f_weight)*cross_train_err + prob.f_weight*1.0
        cond_constrain = float('inf')

        cso = CSO.CSO(prob, cond_constrain=cond_constrain, pop_size=100, max_evaluations=10000,
                      phi=0.05, topology='ring', parallel=parallel)
        sol, ep = cso.evolve()
        processing_time = time.time() - start
        to_print += ep

        selected_features, _ = prob.position_2_solution(sol)
        to_print += ('Selected %d features: ' % len(selected_features)) + ', '.join(
            [str(ele) for ele in selected_features]) + '\n'
        X_train_sel, X_test_sel = X_train[:, selected_features], X_test[:, selected_features]

        clf.fit(X_train_sel, y_train)
        y_test_pred = clf.predict(X_test_sel)
        acc = balanced_accuracy_score(y_true=y_test, y_pred=y_test_pred)
        to_print += 'Acc of selected data: %f\n' % acc
        sel_accs.append(acc)

        f_ratios.append(len(selected_features) / len(sol))
        running_times.append(processing_time)
        to_print += 'Running time: %f\n' % processing_time

    to_print += '====Average Results====\n'
    to_print += 'Average full acc: %f\n' % np.mean(full_test_accs)
    to_print += 'Average selected acc: %f\n' % np.mean(sel_accs)
    to_print += 'Average fratio: %f\n' % np.average(f_ratios)
    to_print += 'Average running time: %f\n' % np.average(running_times)

    f = open(out_dir + str(run) + '.txt', 'w')
    f.write(to_print)
    f.close()
