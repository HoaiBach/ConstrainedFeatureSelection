from scipy import io


def load_data(in_dir, dataset):
    """
    Load dataset from the in_dir where the main data is in in_dir+FSMatlab/dataset.mat and the pre-divided is in
    in_dir+PreMI/dataset.txt
    :param in_dir: directory of input
    :param dataset:
    :return: [(X_train, X_test,y_train, y_test, rel, red, red_cond)]*no_fold, train_test has only 1 fold.
    """

    # read data
    mat = io.loadmat(in_dir + 'FSMatlab/' + dataset + '.mat')
    X = mat['X']
    X = X.astype(float)
    y = mat['Y']
    y = y[:, 0]

    # read pre-split
    f = open(in_dir + 'PreSplit/' + dataset + '.txt', 'r')
    lines = f.readlines()
    if 'Fold' in lines[0]:
        folds_info = []
        for line_idx, line in enumerate(lines):
            if 'Fold' in line:
                train_indices = [int(ele) for ele in lines[line_idx + 1].split(': ')[1].split(', ')]
                test_indices = [int(ele) for ele in lines[line_idx + 2].split(': ')[1].split(', ')]
                X_train, y_train = X[train_indices], y[train_indices]
                X_test, y_test = X[test_indices], y[test_indices]
                folds_info.append((X_train, X_test, y_train, y_test))
        return folds_info
    else:
        train_indices = [int(ele) for ele in lines[0].split(': ')[1].split(', ')]
        test_indices = [int(ele) for ele in lines[1].split(': ')[1].split(', ')]
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]
        return [(X_train, X_test, y_train, y_test)]


if __name__ == '__main__':
    results = load_data(in_dir='/home/nguyenhoai2/Grid/data/', dataset='Sonar_test')
    print('done')
