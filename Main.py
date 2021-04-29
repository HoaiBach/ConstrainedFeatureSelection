import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import balanced_accuracy_score
import time
import CSO
from Utility import DataHandle
from Problem import FeatureSelection

import Orange.preprocess
from Orange.data import Domain, Table

if __name__ == '__main__':
    import sys

    dataset = sys.argv[1]
    run = int(sys.argv[2])
    in_dir = sys.argv[3]
    out_dir = sys.argv[4]
    parallel = sys.argv[5] == 'parallel'
    seed = 1617 * run
    np.random.seed(seed)
    random.seed(seed)

    folds = DataHandle.load_data(in_dir, dataset)
    fold_idx = 1
    to_print = 'Parallel: %s\n' % str(parallel)

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

        # discretizing using Orange
        domain = Domain.from_numpy(X=X_train, Y=y_train)
        table = Table.from_numpy(domain=domain, X=X_train, Y=y_train)
        disc = Orange.preprocess.Discretize(remove_const=False)
        # n_bin = max(min(X_train.shape[0]/3, 10), 2)
        n_bin = len(np.unique(y_train))
        disc.method = Orange.preprocess.EqualWidth(n=n_bin)
        table_dis = disc(table)
        X_train_dis = table_dis.X

        # get info for all features
        to_print += '%d training instances, %d features\n' % X_train.shape
        for clf_name, clf, full_test_acc in zip(clf_names, clfs, full_test_accs):
            if clf_name == 'DT':
                clf.max_depth = min(no_fea/2, np.log2(no_train)-1)
            clf.fit(X_train, y_train)
            y_test_pred = clf.predict(X_test)
            acc = balanced_accuracy_score(y_true=y_test, y_pred=y_test_pred)
            full_test_acc.append(acc)
            to_print += '%s: Full testing accuracy %f\n' % (clf, acc)

        start = time.time()

        if measure.startswith('Wrapper'):
            wrapped_clf = measure.split('-')[1]
            if wrapped_clf == 'KNN':
                evaluator = WrapperMeasure.WrapperMeasureKNN(X=X_train, y=y_train, fratio_weight=0.02, refine=refine)
            elif wrapped_clf == 'SVM':
                evaluator = WrapperMeasure.WrapperMeasureSVM(X=X_train, y=y_train, fratio_weight=0.02, refine=refine)
            elif wrapped_clf == 'DT':
                evaluator = WrapperMeasure.WrapperMeasureDT(X=X_train, y=y_train, fratio_weight=0.02, refine=refine)
            else:
                raise Exception('Classifier %s is not implemented as wrapped classifier!!' % wrapped_clf)
        elif measure in ['MIFS', 'mRMR', 'CIFE', 'JMI', 'FCBF', 'IG']:
            evaluator = InformationTheoryMeasure.InformationMeasureManual(X=X_train_dis, y=y_train, measure_name=measure)
        elif measure == 'Relief':
            evaluator = DistanceMeasure.Relief(X=X_train, y=y_train)
        elif measure == 'FisherScore':
            evaluator = DistanceMeasure.FisherScore(X=X_train, y=y_train)
        elif measure == 'LapScore':
            evaluator = DistanceMeasure.LapScore(X=X_train, y=y_train)
        elif measure == 'TraceRatio':
            evaluator = DistanceMeasure.TraceRatio(X=X_train, y=y_train)
        elif measure == 'InterIntra':
            evaluator = DistanceMeasure.InterIntraDistance(X=X_train, y=y_train)
        elif measure == 'SilhouetteScore':
            evaluator = DistanceMeasure.SilhouetteScore(X=X_train, y=y_train)
        else:
            raise Exception('Measure %s is not implemented!!\n' % measure)
        prob = FeatureSelection(X=X_train, y=y_train, evaluator=evaluator)

        cso = CSO.CSO(prob, pop_size=100, max_evaluations=10000, phi=0.05, topology='ring', parallel=parallel)
        sol, ep = cso.evolve()
        processing_time = time.time() - start
        to_print += ep

        selected_features, _ = prob.position_2_solution(sol)
        to_print += ('Selected %d features: ' % len(selected_features)) + ', '.join(
            [str(ele) for ele in selected_features]) + '\n'
        X_train_sel, X_test_sel = X_train[:, selected_features], X_test[:, selected_features]

        for clf_name, clf, sel_acc in zip(clf_names, clfs, sel_accs):
            if clf_name == 'DT':
                clf.max_depth = min(len(selected_features)/2, np.log2(no_train)-1)
            clf.fit(X_train_sel, y_train)
            y_test_pred = clf.predict(X_test_sel)
            acc = balanced_accuracy_score(y_true=y_test, y_pred=y_test_pred)
            to_print += '%s: Acc of selected data: %f\n' % (clf_name, acc)
            sel_acc.append(acc)

        f_ratios.append(len(selected_features) / len(sol))
        running_times.append(processing_time)
        to_print += 'Running time: %f\n' % processing_time

    to_print += '====Average Results====\n'
    for clf_name, full_acc, sel_acc in zip(clf_names, full_test_accs, sel_accs):
        to_print += '%s: average full acc: %f\n' % (clf_name, np.average(full_acc))
        to_print += '%s: average selected acc: %f\n' % (clf_name, np.average(sel_acc))

    to_print += 'Average fratio: %f\n' % np.average(f_ratios)
    to_print += 'Average running time: %f\n' % np.average(running_times)

    f = open(out_dir + str(run) + '.txt', 'w')
    f.write(to_print)
    f.close()
