"""
Authors: Bach Nguyen
Created: 04/05/2021
Description:
-------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------
"""

import numpy as np


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
            'QsarOralToxicity', 'COIL20', 'ORL', 'Bioresponse', 'RELATHE', 'BASEHOCK', 'Brain1', 'GLIOMA', 'USPS']
            # ,'Gisette', 'TOX-171', 'DLBCL']
runs = 35

methods = ['constrained-single-fit-local-change-n', 'SBPSO', 'VLPSO', 'PSOEMT']
short_methods = ['CCSO', 'SBPSO', 'VLPSO', 'PSOEMT']

in_dir = '/vol/grid-solar/sgeusers/nguyenhoai2/Works/ConstrainedFeatureSelection/Results/Constrain/K=3/'
out_dir = '/ProcessResults/EP/Constrain/'
to_print_acc = 'Dataset\tFull\t' + '\t'.join(short_methods) + '\n'
to_print_ratio = 'Dataset\t' + '\t'.join(short_methods) + '\n'

stand_eval = np.array([])
i = 100
while i <= 10000:
    stand_eval = np.append(stand_eval, i)
    i += 50

for dataset in datasets:
    in_dir_data = in_dir + dataset + '/'
    to_print_acc += dataset
    to_print_ratio += dataset
    add_full_acc = False

    for method, short_method in zip(methods, short_methods):
        accs = np.array([])
        full_accs = np.array([])
        ratios = np.array([])
        for run in range(runs):
            try:
                f = open(in_dir_data + method + '/' + str(run + 1) + '.txt', 'r')
                lines = f.readlines()
                for l_idx, line in enumerate(lines):
                    if line.startswith('Average selected acc'):
                        acc = float(line.split(': ')[1])
                        accs = np.append(accs, acc)
                    elif line.startswith('Average full acc'):
                        full_acc = float(line.split(': ')[1])
                        full_accs = np.append(full_accs, full_acc)
                    elif line.startswith('Average fratio'):
                        ratio = float(line.split(': ')[1])
                        ratios = np.append(ratios, ratio)

            except IOError:
                print(in_dir_data + method + '/' + str(run + 1) + '.txt is not available.')

        if len(accs) == 0:
            accs = np.zeros(runs)
            ratios = np.zeros(runs)

        ave_acc = np.mean(accs)
        ave_fratio = np.mean(ratios)

        if not add_full_acc:
            to_print_acc += '\t %.2f' % (100 * np.mean(full_accs))
            add_full_acc = True
        to_print_acc += '\t %.2f' % (100 * ave_acc)
        to_print_ratio += '\t %4f' % ave_fratio

    to_print_acc += '\n'
    to_print_ratio += '\n'

print('Accuracy')
print(to_print_acc)

print('Ratio')
print(to_print_ratio)
