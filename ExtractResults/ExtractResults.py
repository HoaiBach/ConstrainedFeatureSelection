"""
Authors: Bach Nguyen
Created: 04/05/2021
Description:
-------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------
"""

import numpy as np

datasets = ['Vehicle', 'ImageSegmentation', 'WallRobot', 'German', 'WBCD', 'GuesterPhase', 'Dermatology', 'Ionosphere',
            'Chess', 'Sonar', 'Plant', 'Mice', 'Movementlibras', 'Hillvalley', 'Musk1', 'Semeion', 'LSVT', 'Madelon',
            'Isolet', 'MultipleFeatures', 'Gametes', 'QsarAndrogenReceptor', 'COIL20', 'ORL', 'Yale', 'Bioresponse',
            'Colon', 'SRBCT', 'warpAR10P', 'PCMAC', 'RELATHE', 'BASEHOCK', 'Prostate-GE', 'Carcinom', 'Ovarian',
            'GLI-85']
datasets = ['Bioresponse']
runs = 1
methods = ['constrained-single-fit', 'constrained-single-err', 'constrained-hybrid', 'not_constrained']
short_methods = ['Fit', 'Err', 'Hybrid', 'Std']

in_dir = '/vol/grid-solar/sgeusers/nguyenhoai2/Works/ConstrainedFeatureSelection/Results/LOOCV/'
out_dir = '/home/nguyenhoai2/Research/PycharmProjects/ConstrainedFeatureSelection/ExtractResults/Process/'

to_print_acc = 'Dataset\t' + '\t'.join(short_methods)+'\n'
to_print_fit = 'Dataset\t' + '\t'.join(short_methods)+'\n'

for dataset in datasets:
    in_dir_data = in_dir + dataset +'/'
    to_print_acc += dataset
    to_print_fit += dataset
    for method, short_method in zip(methods, short_methods):
        accs = np.array([])
        fits = np.array([])
        for run in range(runs):
            try:
                f = open(in_dir_data+method+'/'+str(run+1)+'.txt', 'r')
                lines = f.readlines()
                for l_idx, line in enumerate(lines):
                    if line.startswith('Selected'):
                        line = lines[l_idx-7]
                        fit = float(line.split(', ')[0].split(': ')[1])
                        fits = np.append(fits, fit)

                    elif line.startswith('Average selected acc'):
                        acc = float(line.split(': ')[1])
                        accs = np.append(accs, acc)
            except IOError:
                print(in_dir_data+method+'/'+str(run+1)+'.txt is not available.')
        assert len(accs) == runs
        assert len(fits) == runs or len(fits) == 10*runs
        ave_acc = np.mean(accs)
        ave_fit = np.mean(fits)
        to_print_acc += '\t %.2f' % (100*ave_acc)
        to_print_fit += '\t %.4f' % ave_fit
    to_print_acc += '\n'
    to_print_fit += '\n'

print('Accuracy')
print(to_print_acc)

print('Fit')
print(to_print_fit)


