import numpy as np
import Estimators
import pandas as pd


class Slope(Estimators.Estimator):
    """
    The assumption is that hyperparams is a sequence for which
    variance is decreasing.
    """
    def __init__(self,params=None):
        if params is None:
            raise Exception("Slope requires two parameters")

        self.estimator = None
        self.hyperparams = None
        if 'estimator' in params.keys():
            self.estimator = params['estimator']
        if 'hyperparams' in params.keys():
            self.hyperparams = params['hyperparams']
        if 'soften' in params.keys():
            self.soften = params['soften']
        if 'kernel' in params.keys():
            self.kernel = params['kernel']

        if self.estimator is None or self.hyperparams is None:
            raise Exception("Slope requires base estimator and set of hyperparameters")
        
        self.plot = False
        if 'plot' in params.keys():
            self.plot=True

    def estimate(self,target,data):
        means = []
        widths = []
        for h in self.hyperparams: # assumption: hs are ordered ascending
            E = self.estimator(h, self.soften, self.kernel)
            mean = E.estimate(target,data)
            means.append(mean)
            var = E.variance(target,data)
            widths.append(np.sqrt(var))
        intervals = []
        for i in range(len(self.hyperparams)):
            if i < len(self.hyperparams)-1:
                width = max(widths[i], max(widths[i+1:]))
            else:
                width = widths[i]
            intervals.append((means[i] - 2*width, means[i] + 2*width))
            print("[Slope] h = %0.2f, mean = %0.2f, low = %0.2f, high = %0.2f" % (self.hyperparams[i], means[i], intervals[-1][0], intervals[-1][1]), flush=True) 
        index = 0
        curr = [intervals[0][0], intervals[0][1]]
        for i in range(len(intervals)):
            if intervals[i][0] > curr[1] or intervals[i][1] < curr[0]:
                ### Current interval is not overlapping with previous ones, return previous index
                break
            else:
                ### Take intersection
                curr[0] = max(curr[0], intervals[i][0])
                curr[1] = min(curr[1], intervals[i][1])
                index = i
            print("[Slope] curr_low = %0.2f, curr_high = %0.2f" % (curr[0], curr[1]))
        print("[Slope] returning index %d" % (index), flush=True)
        self.means = means
        self.intervals = intervals
        self.index = index
        return means[index]
    
if __name__=='__main__':
    import CCBEnv
    import matplotlib.pyplot as plt
    from Estimators import SmoothedEstimator
    Env = CCBEnv.CCBSimulatedEnv(lip=5,act_dim=1)
    Env.train_logger()
    Env.train_target(100)
    hs = np.logspace(-8,0,9,base=2)
    print(hs)
    n = 1000

    data = Env.gen_logging_data(n)
    estimator=Slope(params={'estimator':SmoothedEstimator,'hyperparams': hs})
    estimator.estimate(Env.target,data)

    print([estimator.intervals[i][1] - estimator.intervals[i][0] for i in range(len(hs))], flush=True)

    errors = np.zeros((2,len(hs)))
    errors[0,:] = [estimator.means[i] - estimator.intervals[i][0] for i in range(len(hs))]
    errors[1,:] = [estimator.intervals[i][1]-estimator.means[i] for i in range(len(hs))]

    plt.errorbar(hs, estimator.means, errors)
    plt.show()
