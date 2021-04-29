from scipy import io
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from skfeature.utility.entropy_estimators import *


if __name__ == '__main__':
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

        f = open(out_dir + dataset + '.txt', 'w')
        f.write(to_write)
        f.close()
