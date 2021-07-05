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

datasets = ['Vehicle', 'WallRobot', 'German', 'GuesterPhase', 'Ionosphere', 'Chess', 'Movementlibras',
            'Hillvalley', 'Musk1', 'Madelon', 'Isolet', 'MultipleFeatures', 'Gametes', 'QsarAndrogenReceptor',
            'QsarOralToxicity', 'COIL20', 'ORL', 'Bioresponse', 'RELATHE', 'BASEHOCK', 'Brain1', 'GLIOMA', 'USPS',
            'Gisette']
# datasets = ['Gisette']
runs = 35

methods = ['not-constrained-n', 'constrained-single-fit-n', 'constrained-single-fit-local-n',
           'constrained-single-fit-local-change-n']
short_methods = ['CSO', 'CCSO', 'CCSO-L', 'CCSO-L-LC']

in_dir = '/Volumes/Data/Work/Research/CurrentResearch/PycharmProjects/CCSO_RawResults/K=3/'
out_dir = '/Volumes/Data/Work/Research/CurrentResearch/PycharmProjects/ConstrainedFeatureSelection/ProcessResults/'

data2method2time = dict()

for dataset in datasets:
    method2time = dict()

    in_dir_data = in_dir + dataset + '/'

    for method, short_method in zip(methods, short_methods):
        times = np.array([])

        for run in range(runs):
            try:
                f = open(in_dir_data + method + '/' + str(run + 1) + '.txt', 'r')
                lines = f.readlines()
                for l_idx, line in enumerate(lines):
                    if line.startswith('Average running time'):
                        time = float(line.split(': ')[1])/60
                        times = np.append(times, time)
            except IOError:
                print(in_dir_data + method + '/' + str(run + 1) + '.txt is not available.')

            if len(times) == 0 or len(times) < runs:
                times = np.zeros(runs)

        method2time[short_method] = times

    data2method2time[dataset] = method2time

# write to table
header = ['Dataset']
header.extend(short_methods)
data = [header]

for dataset in datasets:
    row = [dataset]

    method2time = data2method2time.get(dataset)

    for short_method in short_methods:
        row.append(round(np.mean(method2time.get(short_method)), 2))

    data.append(row)

out_file = open(out_dir+'Latex/Time.txt', 'w')

tabular = tables.Tabular(data)
table = tables.Table(tabular)
table.set_caption('Computation time (in minutes).')
table.set_label('tb:time')
out_file.write(table.as_tex())

out_file.close()