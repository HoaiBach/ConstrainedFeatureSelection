"""
Authors: Bach Nguyen
Created: 02/07/2021
Description:
-------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
"""

import numpy as np
from Utility import DataHandle
from ExtractResults import nonparametric_tests as nontest

def draw_ep(m_eps: dict, title, dir):
    import matplotlib.pyplot as plt
    colors = ['r', 'g', 'b', 'k']
    markers = ['o', 'v', '2', '*']

    for ep, c, m in zip(m_eps.values(), colors, markers):
        plt.plot(list(ep.keys()), list(ep.values()), color=c, marker=m)
    plt.xlabel('# evaluations')
    plt.ylabel('Fitness')
    plt.title(title)
    plt.legend(m_eps.keys())
    plt.savefig('%s%s.pdf' % (dir, title), bbox_inches='tight')
    plt.close()


datasets = ['Vehicle', 'ImageSegmentation', 'WallRobot', 'German', 'WBCD', 'GuesterPhase',
            'Dermatology', 'Ionosphere', 'Chess', 'Sonar', 'Plant', 'Mice',
            'Movementlibras', 'Hillvalley', 'Musk1', 'Semeion', 'LSVT', 'Madelon',
            'Isolet', 'MultipleFeatures', 'Gametes', 'QsarAndrogenReceptor', 'COIL20',
            'ORL', 'Yale', 'Bioresponse', 'Colon', 'SRBCT', 'warpAR10P', 'PCMAC',
            'RELATHE', 'BASEHOCK', 'Prostate-GE', 'Carcinom', 'Ovarian', 'GLI-85',
            'Spect', 'Parkinson', 'LedDisplay', 'Connect4', 'AR10P', 'PIE10P',
            'warpPIE10P', 'lymphoma', 'GLIOMA', 'DLBCL', '9Tumor', 'TOX-171', 'Brain1',
            'ALLAML', 'Leukemia', 'CNS', 'nci9', 'arcene', 'pixraw10P', 'orlraws10P',
            'Leukemia2', 'Leukemia1', 'USPS', 'Gisette']

datasets = ['Vehicle', 'Spect', 'WallRobot', 'German', 'GuesterPhase', 'Ionosphere', 'Chess', 'Movementlibras',
            'Hillvalley', 'Musk1', 'Madelon', 'Isolet', 'MultipleFeatures', 'Gametes', 'QsarAndrogenReceptor',
            'QsarOralToxicity', 'COIL20', 'ORL', 'Bioresponse', 'RELATHE', 'BASEHOCK', 'Brain1', 'GLIOMA', 'USPS',
            'Gisette']
runs = 35
datasets = ['Vehicle']

methods = ['SBPSO', 'VLPSO', 'PSOEMT', 'constrained-single-fit-local-change-n']
short_methods = ['SBPSO', 'VLPSO', 'PSOEMT', 'CCSO']

in_dir = '/vol/grid-solar/sgeusers/nguyenhoai2/Works/ConstrainedFeatureSelection/Results/Constrain/K=3/'
out_dir = '/ExtractResults/EP/Constrain/'
to_print_acc = 'Dataset\tFull\t' + '\t'.join(short_methods) + '\n'
to_print_ratio = 'Dataset\t' + '\t'.join(short_methods) + '\n'

stand_eval = np.array([])
i = 100
while i <= 10000:
    stand_eval = np.append(stand_eval, i)
    i += 50

data2method2acc = dict()
data2method2size = dict()
data2info = dict()
include_full = True

for dataset in datasets:
    method2acc = dict()
    method2size = dict()

    # load data info
    folds = DataHandle.load_data('/vol/grid-solar/sgeusers/nguyenhoai2/Dataset/', dataset)
    X_train, X_test, y_train, y_test = folds[0]
    X = np.append(X_train, X_test, axis=0)
    y = np.append(y_train, y_test)
    noInstances, noFeatures = X.shape
    noClasses = np.unique(y).size
    data2info[dataset] = (noFeatures, noInstances, noClasses)
    if include_full:
        method2size[dataset] = noFeatures

    in_dir_data = in_dir + dataset + '/'

    for method, short_method in zip(methods, short_methods):
        accs = np.array([])
        full_accs = np.array([])
        sizes = np.array([])

        for run in range(runs):
            try:
                f = open(in_dir_data + method + '/' + str(run + 1) + '.txt', 'r')
                lines = f.readlines()
                for l_idx, line in enumerate(lines):
                    if line.startswith('Average selected acc'):
                        acc_str = line.split(': ')[1]
                        if acc_str.isdigit():
                            acc = float(line.split(': ')[1])
                        else:
                            acc = 0
                        accs = np.append(accs, acc)
                    elif line.startswith('Average full acc'):
                        full_acc = float(line.split(': ')[1])
                        full_accs = np.append(full_accs, full_acc)
                    elif line.startswith('Average fratio'):
                        ratio = float(line.split(': ')[1])
                        size = int(ratio*noFeatures)
                        sizes = np.append(sizes, size)
            except IOError:
                print(in_dir_data + method + '/' + str(run + 1) + '.txt is not available.')

            if len(sizes) == 0:
                sizes = np.zeros(runs)

        method2acc[short_method] = accs
        method2size[short_method] = sizes
        if len(full_accs) > 0 and \
                not(short_method in method2acc.keys()) and include_full:
            method2acc['Full'] = full_accs

# Friedman test table vs traditional
benchmarks = ['SBPSO', 'VLPSO', 'PSOEMT']
master = ['CCSO']
friedman_results = []
wdl_scores = [[0.0, 0.0, 0.0] for _ in range(len(benchmarks))]
rank_methods = np.array([0.0]*(len(benchmarks)+1))

for dataset_idx, dataset in enumerate(datasets):
    results = []
    method2acc = data2method2acc.get(dataset)
    for method in benchmarks + master:
        results.append(1.0 - np.array(method2acc(method)))
    _, _, rank, pivots = nontest.friedman_test(*results)
    rank_methods = rank_methods+np.array(rank)
    pivots_dict = {key: pivots[i] for i, key in enumerate(benchmarks+master)}
    compares, _, _, pvalues = nontest. holm_test(pivots_dict, control=master[0])

    pvalues_tmp = [-1.0]*len(benchmarks)
    for compare, pvalue in zip(compares, pvalues):
        method = compare.split(' vs ')[1]
        method_idx = benchmarks.index(method)
        pvalues_tmp[method_idx] = pvalue
    pvalues = pvalues_tmp

    compare_result = [dataset]
    for ben_idx, ben in enumerate(benchmarks):
        if pvalues[ben_idx] < 0.05:
            compare_result.append(round(rank[ben_idx], 2)+'*')
            if rank[ben_idx] > rank[len(benchmarks)]:
                wdl_scores[ben_idx][0] = wdl_scores[ben_idx][0]+1
            else:
                wdl_scores[ben_idx][2] = wdl_scores[ben_idx][2]+1
        else:
            compare_result.append(round(rank[ben_idx], 2))
            wdl_scores[ben_idx][1] = wdl_scores[ben_idx][1]+1

    compare_result.append(round(rank[len(benchmarks)], 2))

    friedman_results.append(compare_result)
print('test')
