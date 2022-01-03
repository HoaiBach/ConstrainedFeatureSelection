"""
Authors: Bach Nguyen
Created: 02/07/2021
Description:
-------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------
"""

import numpy as np
from Utility import DataHandle
from ProcessResults import nonparametric_tests as nontest
from ProcessResults import tables


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

datasets = ['Vehicle', 'WallRobot', 'German', 'GuesterPhase', 'Ionosphere', 'Chess', 'Movementlibras',
            'Hillvalley', 'Musk1', 'Madelon', 'Isolet', 'MultipleFeatures', 'Gametes', 'QsarAndrogenReceptor',
            'QsarOralToxicity', 'COIL20', 'ORL', 'Bioresponse', 'RELATHE', 'BASEHOCK', 'Brain1', 'GLIOMA', 'USPS',
            'Gisette']

datasets = ['Vehicle', 'WallRobot', 'German', 'GuesterPhase', 'Ionosphere', 'Chess', 'Movementlibras',
            'Hillvalley', 'Musk1', 'Madelon', 'Isolet', 'MultipleFeatures', 'Gametes', 'QsarAndrogenReceptor',
            'QsarOralToxicity', 'COIL20', 'ORL', 'Bioresponse', 'RELATHE', 'BASEHOCK', 'GLIOMA', 'USPS',
            'Gisette']

# datasets = ['Gisette']
runs = 35
include_full = False

# methods = ['CFS1', 'mRMR1', 'reliefF1', 'RFS1', 'GFS1', 'SDFS1', 'TKFS1',  'constrained-single-fit-local-change-n']
# short_methods = ['CFS', 'mRMR', 'reliefF', 'RFS', 'GFS', 'SDFS', 'TKFS', 'CCSO']
methods = ['constrained-single-fit-n', 'constrained-single-fit-local-n', 'constrained-single-fit-not-local-change-n',
           'constrained-single-fit-local-change-n']
short_methods = ['C', 'CL', 'CS', 'CLS']
if include_full:
    benchmarks = ['Full'].extend(short_methods[:len(short_methods)-1])
else:
    benchmarks = short_methods[:len(short_methods)-1]
master = 'CLS'
combine = np.append(benchmarks, master)

out_dir = '/home/nguyenhoai2/Research/PycharmProjects/ConstrainedFeatureSelection/ProcessResults/'
data_dir = '/vol/grid-solar/sgeusers/nguyenhoai2/Dataset/'
in_dir = '/local/scratch/ConstrainedFeatureSelection/K=3/'

stand_eval = np.array([])
i = 100
while i <= 10000:
    stand_eval = np.append(stand_eval, i)
    i += 50

data2method2acc = dict()
data2method2size = dict()
data2info = dict()

for dataset in datasets:
    method2acc = dict()
    method2size = dict()

    # load data info
    folds = DataHandle.load_data(data_dir, dataset)
    X_train, X_test, y_train, y_test = folds[0]
    X = np.append(X_train, X_test, axis=0)
    y = np.append(y_train, y_test)
    noInstances, noFeatures = X.shape
    noClasses = np.unique(y).size
    data2info[dataset] = (noFeatures, noInstances, noClasses)
    if include_full:
        method2size['Full'] = np.array([noFeatures]*runs)

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
                        acc = float(line.split(': ')[1])
                        if np.isnan(acc):
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
                accs = np.append(accs, 0.0)
                sizes = np.append(sizes, 0)
                print(in_dir_data + method + '/' + str(run + 1) + '.txt is not available.')

            if len(sizes) == 0:
                sizes = np.zeros(runs)
            if len(accs) == 0:
                accs = np.zeros(runs)

        method2acc[short_method] = accs
        method2size[short_method] = sizes
        if len(full_accs) > 0 and \
                not('Full' in method2acc.keys()) and include_full:
            method2acc['Full'] = full_accs

        data2method2acc[dataset] = method2acc
        data2method2size[dataset] = method2size

# Friedman test table vs traditional

friedman_results = []
wdl_scores = [[0.0, 0.0, 0.0] for _ in range(len(benchmarks))]
ranks = np.array([0.0] * (len(combine)))

for dataset_idx, dataset in enumerate(datasets):
    results = []
    method2acc = data2method2acc.get(dataset)
    for method in combine:
        results.append(1.0 - np.array(method2acc.get(method)))
    _, _, rank, pivots = nontest.friedman_test(*results)
    ranks = ranks + np.array(rank)
    pivots_dict = {key: pivots[i] for i, key in enumerate(combine)}
    compares, _, _, pvalues = nontest. holm_test(pivots_dict, control=master)

    pvalues_tmp = [-1.0]*len(benchmarks)
    for compare, pvalue in zip(compares, pvalues):
        method = compare.split(' vs ')[1]
        method_idx = benchmarks.index(method)
        pvalues_tmp[method_idx] = pvalue
    pvalues = pvalues_tmp

    compare_result = [dataset]
    for ben_idx, ben in enumerate(benchmarks):
        if pvalues[ben_idx] < 0.05:
            compare_result.append(str(round(rank[ben_idx], 2)) + '*')
            if rank[ben_idx] > rank[len(benchmarks)]:
                wdl_scores[ben_idx][0] = wdl_scores[ben_idx][0]+1
            else:
                wdl_scores[ben_idx][2] = wdl_scores[ben_idx][2]+1
        else:
            compare_result.append(round(rank[ben_idx], 2))
            wdl_scores[ben_idx][1] = wdl_scores[ben_idx][1]+1

    compare_result.append(round(rank[len(benchmarks)], 2))

    friedman_results.append(compare_result)

ranks = ranks / len(datasets)

# write to table
header = ['Dataset']
header.extend(combine)
data_nf = [header]
data_acc = [header]

for dataset in datasets:
    row_nf = [dataset]
    row_acc = [dataset]

    methods_nf = data2method2size.get(dataset)
    methods_acc = data2method2acc.get(dataset)

    for short_method in combine:
        row_nf.append(round(np.mean(methods_nf.get(short_method)), 2))
        row_acc.append(round(np.mean(methods_acc.get(short_method))*100, 2))

    data_nf.append(row_nf)
    data_acc.append(row_acc)

test_row = ['Test']
for idx, benchmarks in enumerate(benchmarks):
    wdl = wdl_scores[idx]
    test_row.append('/'.join([str(ele) for ele in wdl]))
test_row.append('N/A')
data_acc.append(test_row)

rank_row = ['Rank']
for rank in ranks:
    rank_row.append(str(round(rank, 2)))
data_acc.append(rank_row)

out_file = open(out_dir+'Latex/Output.txt', 'w')

tabular = tables.Tabular(data_nf)
table = tables.Table(tabular)
table.set_caption('Number of selected features')
table.set_label('tb:nf')
out_file.write(table.as_tex())

tabular = tables.Tabular(data_acc)
table = tables.Table(tabular)
table.set_caption('Testing accuracies')
table.set_label('tb:acc')
out_file.write(table.as_tex()+'\n\n')

out_file.close()


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