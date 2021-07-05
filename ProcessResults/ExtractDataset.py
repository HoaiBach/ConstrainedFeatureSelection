import numpy as np
from scipy import io

datasets = ['Vehicle', 'WallRobot', 'German', 'GuesterPhase', 'Ionosphere', 'Chess', 'Movementlibras',
            'Hillvalley', 'Musk1', 'Madelon', 'Isolet', 'MultipleFeatures', 'Gametes', 'QsarAndrogenReceptor',
            'QsarOralToxicity', 'COIL20', 'ORL', 'Bioresponse', 'RELATHE', 'BASEHOCK', 'Brain1', 'GLIOMA', 'USPS',
            'Gisette']


nis = []
nfs = []
ncs = []
for dataset in datasets:
    mat = io.loadmat('/Volumes/Data/Work/Research/CurrentResearch/Datasets/FSMatlab/'+dataset+'.mat')
    X = mat['X']    # data
    X = X.astype(float)
    y = mat['Y']    # label
    y = y[:, 0]
    ni, nf = X.shape
    nc = len(np.unique(y))
    nis.append(ni)
    nfs.append(nf)
    ncs.append(nc)
sorted = np.argsort(nfs)
for idx in sorted:
    print('%s & %d & %d & %d \\\\' % (datasets[idx], nfs[idx], nis[idx], ncs[idx]))
datasets = np.array(datasets)[sorted]
print('[ '+', '.join(['\''+dataset+'\'' for dataset in datasets])+' ]')
    # for o, d in zip(ori, des):
    #     os.rename(dir+dataset+'/'+o, dir+dataset+'/'+d)
