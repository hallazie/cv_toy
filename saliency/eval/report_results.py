
import numpy as np


#eval = np.load('./fastsal5.npy')

eval = np.load('resnet_pruned_075.npy')

for e in eval:
    NSS = []
    AUC_Judd = []
    AUC_Borji = []
    sAUC = []
    CC = []
    SIM = []
    KL = []
    IG = []
    mNSS = []
    fname = []

    d = e['scores']
    NSS = NSS + d['NSS']
    mNSS.append(np.mean(d['NSS']))
    fname.append(e['model'])
    AUC_Judd = AUC_Judd + d['AUC_Judd']
    AUC_Borji = AUC_Borji + d['AUC_Borji']
    sAUC = sAUC + d['sAUC']
    CC = CC + d['CC']
    SIM = SIM + d['SIM']
    KL = KL + d['KL']

    #IG = IG + d['IG']
    print(fname)
    print('NSS {}'.format(np.mean(NSS)))
    print('AUC Judd {}'.format(np.mean(AUC_Judd)))
    print('AUC Borji {}'.format(np.mean(AUC_Borji)))
    print('sAUC {}'.format(np.mean(sAUC)))
    print('CC {}'.format(np.mean(CC)))
    print('SIM {}'.format(np.mean(SIM)))
    print('KL {}'.format(np.mean(KL)))
#print('IG {}'.format(np.mean(IG)))
