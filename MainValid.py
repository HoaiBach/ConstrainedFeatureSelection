import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import balanced_accuracy_score
import time
import CSOValid
from Utility import DataHandle
from Problem import FeatureSelection
import WorldPara
from Utility import Helpers


if __name__ == '__main__':
    import sys

    dataset = sys.argv[1]
    run = int(sys.argv[2])
    in_dir = sys.argv[3]
    out_dir = sys.argv[4]
    parallel = sys.argv[5] == 'parallel'
    if sys.argv[6].startswith('constrained'):
        WorldPara.PENALISE_WORSE_THAN_FULL = True
        WorldPara.ENHANCE_CONSTRAIN = sys.argv[6].split('-')[1] == 'enhance'
        WorldPara.ERR_CONSTRAIN = sys.argv[6].split('-')[2] == 'error'
    else:
        WorldPara.PENALISE_WORSE_THAN_FULL = False

    if WorldPara.ENHANCE_CONSTRAIN:
        assert WorldPara.PENALISE_WORSE_THAN_FULL

    seed = 1617 * run
    np.random.seed(seed)
    random.seed(seed)

    folds = DataHandle.load_data(in_dir, dataset)
    fold_idx = 1
    to_print = 'Parallel: %s\n' % str(parallel)
    to_print += 'Constrained: %s\n' % str(WorldPara.PENALISE_WORSE_THAN_FULL)
    to_print += 'Enhance: %s\n' % str(WorldPara.ENHANCE_CONSTRAIN)
    to_print += 'Error Constrain: %s\n' % str(WorldPara.ERR_CONSTRAIN)

    full_test_accs = []
    sel_accs = []
    f_ratios = []
    sel_accs_valid = []
    f_ratios_valid = []
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

        cross_train_err = Helpers.kFoldCrossValidation(X_train, y_train, prob.clf, prob.skf)
        if WorldPara.ERR_CONSTRAIN:
            cond_constrain = cross_train_err
        else:
            cond_constrain = (1.0-prob.f_weight)*cross_train_err + prob.f_weight*1.0

        cso = CSOValid.CSO(prob, cond_constrain=cond_constrain, pop_size=100, max_evaluations=10000,
                      phi=0.05, topology='ring', parallel=parallel)
        sol, ep, sol_valid = cso.evolve()
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

        selected_features, _ = prob.position_2_solution(sol_valid)
        to_print += ('Selected %d features (valid): ' % len(selected_features)) + ', '.join(
            [str(ele) for ele in selected_features]) + '\n'
        X_train_sel, X_test_sel = X_train[:, selected_features], X_test[:, selected_features]

        clf.fit(X_train_sel, y_train)
        y_test_pred = clf.predict(X_test_sel)
        acc = balanced_accuracy_score(y_true=y_test, y_pred=y_test_pred)
        to_print += 'Acc of selected data (valid): %f\n' % acc
        sel_accs_valid.append(acc)

        f_ratios_valid.append(len(selected_features) / len(sol))

        running_times.append(processing_time)
        to_print += 'Running time: %f\n' % processing_time

    to_print += '====Average Results====\n'
    to_print += 'Average full acc: %f\n' % np.mean(full_test_accs)
    to_print += 'Average selected acc: %f\n' % np.mean(sel_accs)
    to_print += 'Average fratio: %f\n' % np.average(f_ratios)
    to_print += 'Average selected acc (valid): %f\n' % np.mean(sel_accs_valid)
    to_print += 'Average fratio (valid): %f\n' % np.average(f_ratios_valid)
    to_print += 'Average running time: %f\n' % np.average(running_times)

    f = open(out_dir + str(run) + '.txt', 'w')
    f.write(to_print)
    f.close()
