import json
import numpy as np
import matplotlib.pyplot as plt
import copy

commands = [0,1,2,3,4,5,6,7,8,9]
iters = 100

data_dict = {
    'Slope': [],
    '0.25': [],
    '0.03125': [],
    }

K = [
'Slope', 
'0.25', 
'0.03125']
ns = []

data = {
    
    }

for c in commands:
    for i in range(1,iters+1):
        x = json.loads(open('./results/command_num=%d_replicate=%d.json' % (c, i), 'r').readlines()[0])
        n = x['samples']
        if n not in data.keys():
            data[n] = copy.deepcopy(data_dict)
            ns.append(n)
        for k in K:
            data[n][k].append(x[k])

ns.sort()

fig = plt.figure()
ls = []
for k in K:
    x = np.array([np.mean(data[n][k]) for n in ns])
    z = np.array([np.std(data[n][k]) for n in ns])
    ls.append(plt.plot(ns, x,linewidth=2))
    plt.fill_between(ns, x - 2/np.sqrt(iters)*z, x + 2/np.sqrt(iters)*z,alpha=0.2)

ax = plt.gca()
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('MSE')
ax.set_xlabel('Number of samples')

K[K.index('Slope')] = 'Slope'
plt.legend(K)

plt.savefig('./cb_learning_curve.pdf', format='pdf', dpi=100,bbox_inches='tight')
# plt.show()
