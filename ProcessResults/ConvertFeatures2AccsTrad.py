import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import balanced_accuracy_score
from Utility import DataHandle
import os
import WorldPara

if __name__ == '__main__':

    datasets = ['Vehicle', 'Spect', 'WallRobot', 'German', 'GuesterPhase', 'Ionosphere', 'Chess', 'Movementlibras',
                'Hillvalley', 'Musk1', 'Madelon', 'Isolet', 'MultipleFeatures', 'Gametes', 'QsarAndrogenReceptor',
                'QsarOralToxicity', 'COIL20', 'ORL', 'Bioresponse', 'RELATHE', 'BASEHOCK', 'Brain1', 'GLIOMA', 'USPS',
                'Gisette']
    algs = ['CFS', 'GFS', 'mRMR', 'reliefF', 'RFS', 'TKFS']
    datasets = ['MultipleFeatures']
    algs = ['CFS']

    for dataset in datasets:
        # print(dataset)
        folds = DataHandle.load_data('/vol/grid-solar/sgeusers/nguyenhoai2/Dataset/', dataset)
        for alg in algs:
            # print('-----%s' % alg)
            in_dir = '/vol/grid-solar/sgeusers/nguyenhoai2/Works/ConstrainedFeatureSelection/Results/Constrain/K=3/'\
                     + dataset+'/'+alg+'/'

            to_write = ''
            accs = []
            try:
                f = open(in_dir+'1.txt', 'r')
                lines = f.readlines()
                for l_idx, line in enumerate(lines):
                    if 'Fold' in line:
                        to_write += line
                        fold_idx = line.split(' ')[1]
                        fold_idx = int(fold_idx[0])-1

                        # extracting selected features
                        while not lines[l_idx].startswith('Indices'):
                            l_idx += 1
                        sel_indices = lines[l_idx].split(':')[1]
                        sel_indices = sel_indices.strip()
                        sel_indices = [int(ele) for ele in sel_indices.split(' ')]
                        sel_indices = np.array(sel_indices)
                        if alg == 'TKFS':
                            sel_indices = sel_indices-1
                        to_write += 'Selected %d features: %s\n' % (len(sel_indices), ', '.join([str(ele) for ele in
                                                                                              sel_indices]))

                        # calculate accuracy
                        X_train, X_test, y_train, y_test = folds[fold_idx]
                        no_train, no_fea = X_train.shape
                        x_max = np.max(X_train, axis=0)
                        x_min = np.min(X_train, axis=0)
                        X_train = (X_train - x_min) / (x_max - x_min)
                        X_test = (X_test - x_min) / (x_max - x_min)
                        X_train = np.nan_to_num(X_train)
                        X_test = np.nan_to_num(X_test)
                        X_train_sel = X_train[:, sel_indices]
                        X_test_sel = X_test[:, sel_indices]
                        clf = KNN(n_neighbors=WorldPara.NUM_NEIGHBORS)
                        clf.fit(X_train_sel, y_train)
                        y_test_pred = clf.predict(X_test_sel)
                        acc = balanced_accuracy_score(y_true=y_test, y_pred=y_test_pred)
                        accs.append(acc)
                        to_write += 'Acc of selected data %s\n' % acc
                        l_idx += 1
            except IOError:
                print('%s does not have the results of %s.' % (dataset, alg))

            to_write += '====Average Results====\n'
            to_write += 'Average selected acc: %s\n' %(np.mean(accs))

            no_runs = 35
            out_dir = '/vol/grid-solar/sgeusers/nguyenhoai2/Works/ConstrainedFeatureSelection/' \
                      'Results/Constrain/K=3/%s/%s1/' % (dataset, alg)
            os.makedirs(out_dir, exist_ok=True)
            for run in range(no_runs):
                f = open(out_dir + str(run+1) + '.txt', 'w')
                f.write(to_write)
                f.close()
