import numpy as np
import random
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import balanced_accuracy_score
import time
import CSO
from Utility import DataHandle
from Problem import FeatureSelection
import WorldPara

if __name__ == '__main__':
    import sys

    dataset = sys.argv[1]
    run = int(sys.argv[2])
    in_dir = sys.argv[3]  # where is the FSmatlab
    out_dir = sys.argv[4]  # where to write run.txt
    parallel = sys.argv[5] == 'parallel'  # parallel or not

    # Parameter for Constrain optimisation
    if sys.argv[6].startswith('constrained'):
        # not_constrained/constrained-single-fit/constrained-single-err/constrained-hybrid
        splits = sys.argv[6].split('-')
        WorldPara.CONSTRAIN_MODE = splits[1]
        if WorldPara.CONSTRAIN_MODE == 'single':
            WorldPara.CONSTRAIN_TYPE = splits[2]
        elif WorldPara.CONSTRAIN_MODE == 'hybrid':
            WorldPara.CONSTRAIN_TYPE = 'err'
        else:
            raise Exception('%s constrain mode is not implemented!' % WorldPara.CONSTRAIN_MODE)
    elif sys.argv[6] == 'not-constrained':
        WorldPara.CONSTRAIN_MODE = None
    else:
        raise Exception('%s constrain mode is not implemented!' % sys.argv[6])

    # Parameter for local search
    if sys.argv[7].startswith('local'):
        WorldPara.LOCAL_SEARCH = True
        splits = sys.argv[7].split('-')
        if splits[1] == 'asym':
            WorldPara.LOCAL_TYPE = 'asym'
            WorldPara.LOCAL_ASYM_FLIP = int(splits[2])
            WorldPara.LOCAL_STUCK_THRESHOLD = int(splits[3])
        elif splits[1] == 'std':
            WorldPara.LOCAL_TYPE = 'std'
            WorldPara.LOCAL_STUCK_THRESHOLD = int(splits[2])
        else:
            raise Exception('Local search %s is not implemented!' % WorldPara.LOCAL_TYPE)
    elif sys.argv[7].startswith('not-local'):
        WorldPara.LOCAL_SEARCH = False
    else:
        raise Exception('%s local mode is not implemented!' % sys.argv[7])

    # Parameter for length change
    if sys.argv[8].startswith('change'):
        WorldPara.LENGTH_UPDATE = True
        WorldPara.LENGTH_STUCK_THRESHOLD = int(sys.argv[8].split('-')[1])
    elif sys.argv[8] == 'not-change':
        WorldPara.LENGTH_UPDATE = False
    else:
        raise Exception('%s length change is not implemented!' % sys.argv[8])

    # parameter for surrogate model
    splits = sys.argv[9].split('-')
    WorldPara.SURROGATE_VERSION = splits[0]
    WorldPara.SURROGATE_UPDATE_DURATION = int(splits[1])

    # check constraints
    assert WorldPara.LOCAL_STUCK_THRESHOLD < WorldPara.LENGTH_STUCK_THRESHOLD

    seed = 1617 * run
    np.random.seed(seed)
    random.seed(seed)

    to_print = 'Number neighbors: %d\n' % WorldPara.NUM_NEIGHBORS
    to_print += 'Parallel: %s\n' % str(parallel)
    to_print += 'Constrain mode: %s\n' % str(WorldPara.CONSTRAIN_MODE)
    if WorldPara.CONSTRAIN_MODE == 'single':
        to_print += '\tConstrain type: %s\n' % str(WorldPara.CONSTRAIN_TYPE)
    to_print += 'Local search: %s\n' % str(WorldPara.LOCAL_SEARCH)
    to_print += '\tStuck Threshold: %d\n' % WorldPara.LOCAL_STUCK_THRESHOLD
    to_print += '\tLocal search iterations: %d\n' % WorldPara.LOCAL_ITERATIONS
    to_print += '\tProportion of population: %f\n' % WorldPara.TOP_POP_RATE
    to_print += '\tLocal search type: %s\n' % WorldPara.LOCAL_TYPE
    if WorldPara.LOCAL_SEARCH and WorldPara.LOCAL_TYPE == 'asym':
        to_print += '\tLocal search flip: %d\n' % WorldPara.LOCAL_ASYM_FLIP
    to_print += 'Length change: %s\n' % str(WorldPara.LENGTH_UPDATE)
    to_print += '\tLength Stuck Threshold: %d\n' % WorldPara.LENGTH_STUCK_THRESHOLD
    to_print += '\tLength search iterations: %d\n' % WorldPara.LENGTH_ITERATIONS
    to_print += 'Surrogate model\n'
    to_print += '\t Surrogate version: %s\n' % WorldPara.SURROGATE_VERSION
    to_print += '\t Surrogate update duration: %s\n' % WorldPara.SURROGATE_UPDATE_DURATION

    full_test_accs = []
    sel_accs = []
    f_ratios = []
    running_times = []
    clf = KNN(n_neighbors=WorldPara.NUM_NEIGHBORS)

    folds = DataHandle.load_data(in_dir, dataset)
    fold_idx = 1
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
                                init_style='Random', fratio_weight=0.02)

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
