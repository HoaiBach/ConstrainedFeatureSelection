"""
Authors: Bach Nguyen
Created: 04/05/2021
Description:
-------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------
"""

import numpy as np
from ProcessResults import Visualisation

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

in_dir = '/Volumes/Data/Work/Research/CurrentResearch/PycharmProjects/CCSO_RawResults/K=3/'
out_dir = '/Volumes/Data/Work/Research/CurrentResearch/Latex/ConstrainedFeatureSelection/Figures/constrain/'

datasets = ['Vehicle', 'Spect', 'WallRobot', 'German', 'GuesterPhase', 'Ionosphere', 'Chess', 'Movementlibras',
            'Hillvalley', 'Musk1', 'Madelon', 'Isolet', 'MultipleFeatures', 'Gametes', 'QsarAndrogenReceptor',
            'QsarOralToxicity', 'COIL20', 'ORL', 'Bioresponse', 'RELATHE', 'BASEHOCK', 'Brain1', 'GLIOMA', 'USPS']
            # , 'Gisette']
# datasets = ['Vehicle', 'Brain1']

runs = 35

# methods = [ 'constrained-single-fit-n', 'constrained-single-fit-local-n']
# short_methods = ['CCSO', 'CCSO-L']
# out_dir = './Figures/Local/'
# methods = ['not-constrained-n', 'constrained-single-fit-n']
# short_methods = ['CSO', 'CCSO']
# out_dir = './Figures/Constrain/'
methods = ['constrained-single-fit-local-n', 'constrained-single-fit-local-change-n']
short_methods = ['CCSO-L', 'CCSO-L-LC']
out_dir = './Figures/Length/'

stand_eval = np.array([])
i = 100
while i <= 10000:
    stand_eval = np.append(stand_eval, i)
    i += 50

for dataset in datasets:
    in_dir_data = in_dir + dataset + '/'
    fold2alg2eval2fit = dict()

    for method, short_method in zip(methods, short_methods):
        fold2eval2fit = dict()
        for run in range(runs):
            try:
                f = open(in_dir_data + method + '/' + str(run + 1) + '.txt', 'r')
                lines = f.readlines()
                fold_idx = -1
                current_eval = -1

                for l_idx, line in enumerate(lines):
                    if 'Fold' in line:
                        split = line.split(' ')[1]
                        fold_idx = split.split('=')[0]
                        if not(fold_idx in fold2eval2fit.keys()):
                            fold2eval2fit[fold_idx] = dict()
                        current_eval = -1
                    if line.startswith('At'):
                        splits = lines[l_idx].split(': ')
                        no_eval = int(splits[0].split(' ')[1])
                        no_eval = int(no_eval/50)*50

                        while no_eval > current_eval + 50 and not (current_eval < 0) and not (current_eval+50 > 10000):
                            # jump step
                            fold2eval2fit.get(fold_idx)[current_eval+50] = fold2eval2fit.get(fold_idx)[current_eval]
                            current_eval += 50

                        fit = float(splits[1].split(', ')[0])
                        if no_eval <= 10000:
                            if not (no_eval in fold2eval2fit.get(fold_idx).keys()):
                                fold2eval2fit.get(fold_idx)[no_eval] = fit
                            else:
                                fold2eval2fit.get(fold_idx)[no_eval] += fit
                            current_eval = no_eval
                        else:
                            print(no_eval)
                            print(in_dir_data + method + '/' + str(run + 1) + '.txt')
                            break
            except IOError:
                print(in_dir_data + method + '/' + str(run + 1) + '.txt is not available.')

        # several tests should be passed
        # check the number of evaluations should be correct
        for eval2fit in fold2eval2fit.values():
            evals = np.array([ele for ele in eval2fit.keys()])
            dif = np.abs(np.sum(evals-stand_eval))
            assert dif == 0
        # check the fit should not be increased
        increase = False
        for eval2fit in fold2eval2fit.values():
            fits = np.array([ele for ele in eval2fit.values()])
            for idx in np.arange(0, len(fits)-1):
                if fits[idx] < fits[idx+1]:
                    increase = True
                    break
        assert not increase

        for fold, eval2fit in fold2eval2fit.items():
            # fix the eval2fit first to fit in runs
            for eval, value in eval2fit.items():
                fold2eval2fit[fold][eval] = value/runs

            if not(fold in fold2alg2eval2fit.keys()):
                fold2alg2eval2fit[fold] = dict()
            fold2alg2eval2fit[fold][short_method] = eval2fit

    for fold, alg2eval2fit in fold2alg2eval2fit.items():
        Visualisation.drawLineGraphs(data=alg2eval2fit, caption=dataset,
                                     out_dir=out_dir+'%s-%s.pdf' % (dataset, fold),
                                     x_label='NoEvals', y_label='Fitness')
