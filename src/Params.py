import numpy as np

# hs = [0.25,0.03125]
hs = [x for x in np.logspace(1, 7, num=7, base=0.5)]
# hs = [x for x in np.linspace(0.01, 0.5, 20)]
hs.reverse()
#hs = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]

FRIENDLY = {'method': 'friendly', 'alpha': 0.7, 'beta': 0.2, 'l': 10}
ADVERSARIAL = {'method': 'adversarial', 'alpha': 0.3, 'beta': 0.2, 'l': 10}
NEUTRAL = {'method': 'neutral', 'alpha': 0., 'beta': 0., 'l': 10}


ns = [10, 30, 100, 300, 1000, 3000]
Replicates = 100
feat_dim=10
act_dim=2
lip=10

if __name__=='__main__':
    ### Generate a huge bash file of all the commands
    for n in ns:
        print("python3 Experiment2.py --start_iter 1 --total_iter %d --samples %d --feat_dim %d --act_dim %d --lip %d" % (Replicates, n, feat_dim, act_dim-1, lip))
