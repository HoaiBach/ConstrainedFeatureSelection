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


datasets = ['Vehicle', 'ImageSegmentation', 'WallRobot', 'German', 'WBCD', 'GuesterPhase', 'Dermatology', 'Ionosphere',
            'Chess', 'Sonar', 'Plant', 'Mice', 'Movementlibras', 'Hillvalley', 'Musk1', 'Semeion', 'LSVT', 'Madelon',
            'Isolet', 'MultipleFeatures', 'Gametes', 'QsarAndrogenReceptor', 'COIL20', 'ORL', 'Yale', 'Bioresponse',
            'Colon', 'SRBCT', 'warpAR10P', 'PCMAC', 'RELATHE', 'BASEHOCK', 'Prostate-GE', 'Carcinom', 'Ovarian',
            'GLI-85', 'USPS']
datasets = ['Vehicle', 'ImageSegmentation', 'German', 'WBCD', 'Dermatology', 'Ionosphere',
            'Chess', 'Sonar', 'Plant', 'Mice', 'Movementlibras', 'Hillvalley', 'Musk1', 'Semeion', 'LSVT', 'Madelon',
            'Isolet', 'MultipleFeatures', 'Gametes', 'QsarAndrogenReceptor', 'COIL20', 'ORL', 'Yale',
            'Colon', 'SRBCT', 'warpAR10P', 'RELATHE',  'Prostate-GE', 'Carcinom', 'GLI-85']
runs = 5
methods = ['constrained-single-fit', 'not-constrained']
short_methods = ['con', 'not']

in_dir = '/vol/grid-solar/sgeusers/nguyenhoai2/Works/ConstrainedFeatureSelection/Results/Constrain/K=3/'
out_dir = '/home/nguyenhoai2/Research/PycharmProjects/ConstrainedFeatureSelection/ExtractResults/EP/Constrain/'
to_print_acc = 'Dataset\tFull\t' + '\t'.join(short_methods) + '\n'
to_print_fit = 'Dataset\t' + '\t'.join(short_methods) + '\n'
to_print_ratio = 'Dataset\t' + '\t'.join(short_methods) + '\n'

stand_eval = np.array([])
i = 100
while i <= 10000:
    stand_eval = np.append(stand_eval, i)
    i += 50

for dataset in datasets:
    in_dir_data = in_dir + dataset + '/'
    to_print_acc += dataset
    to_print_fit += dataset
    to_print_ratio += dataset
    add_full_acc = False
    m_ep = dict()

    for method, short_method in zip(methods, short_methods):
        accs = np.array([])
        full_accs = np.array([])
        fits = np.array([])
        ratios = np.array([])
        ep = dict()
        for run in range(runs):
            try:
                f = open(in_dir_data + method + '/' + str(run + 1) + '.txt', 'r')
                lines = f.readlines()
                for l_idx, line in enumerate(lines):
                    if 'Fold 1=' in line:
                        l_idx += 4

                        if (run+1) == 1:
                            while lines[l_idx].startswith('At'):
                                splits = lines[l_idx].split(': ')
                                no_eval = int(splits[0].split(' ')[1])
                                # diff_eval = np.abs(stand_eval-no_eval)
                                # closest_idx = np.argmin(diff_eval)
                                # print('%d vs %d' % (no_eval, stand_eval[closest_idx]))
                                fit = float(splits[1].split(', ')[0])
                                ep.update({no_eval: fit})
                                l_idx += 7

                    if line.startswith('Selected'):
                        line = lines[l_idx - 7]
                        fit = float(line.split(', ')[0].split(': ')[1])
                        fits = np.append(fits, fit)

                    elif line.startswith('Average selected acc'):
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

        ave_acc = np.mean(accs)
        ave_fit = np.mean(fits)
        ave_fratio = np.mean(ratios)

        if not add_full_acc:
            to_print_acc += '\t %.2f' % (100 * np.mean(full_accs))
            add_full_acc = True
        to_print_acc += '\t %.2f' % (100 * ave_acc)
        to_print_fit += '\t %.4f' % ave_fit
        to_print_ratio += '\t %4f' % ave_fratio

        m_ep.update({short_method: ep})
    to_print_acc += '\n'
    to_print_fit += '\n'
    to_print_ratio += '\n'

    # draw evolutionary process
    draw_ep(m_ep, title=dataset, dir=out_dir)

print('Accuracy')
print(to_print_acc)

print('Fit')
print(to_print_fit)

print('Ratio')
print(to_print_ratio)
