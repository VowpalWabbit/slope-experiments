import numpy as np

class Estimator(object):

    def __init__(self,params=None):
        self.params=params

    def estimate(self,target,data):
        return 0
    
    def variance(self,target,data):
        return 0


class SmoothedEstimator(Estimator):
    def __init__(self,h, soften=False, kernel = None):
        super(Estimator).__init__()
        self.h = h
        self.soften = soften
        self.kernel = kernel
        
    def epanechnikov_kernel(self, u):
        return 0.75*(1-u**2)*(1 if abs(u) <= 1 else 0)

    def epanechnikov_int(self,lo,hi):
        '''
        :return: Definite integral of the kernel from between lo and hi. Assumes that they are within bounds.
        '''
        return 0.75*(hi-hi**3/3.0) - 0.75*(lo-lo**3/3.0)
    
    def boxcar_kernel(self, u):
        return 0.5*(1 if abs(u) <= 1 else 0)
    
    def boxcar_int(self,lo,hi):
        '''
        :return: Definite integral of the kernel from between lo and hi. Assumes that they are within bounds.
        '''
        return abs(hi -lo)/2.
    
    def get_density(self,a,p):
        if a >= p['tau_low'] and a <= p['tau_high']:
            return p['rho_i']
        else:
            return p['rho_o']
        
    def estimate(self,target,data):
        val = 0
        rewards = []
        for tup in data:
            (x,a,l,p) = tup
            target_action = target.get_action(x)['action']
            if self.kernel == None:
                # boundary bias not handled here, experiment at your own risk
                if np.all(np.abs(a - target_action) <= self.h):
                    val += l/self.get_prob(target_action, p)
            elif self.kernel == "epanechnikov":
                den = 1.
                num = l
                for d in range(target_action.shape[1]): 
                    delta = (a[0][d] - target_action[0,d])/self.h
                    num *= self.epanechnikov_kernel(delta)
                    t_lo = max(target_action[0,d]-self.h, -1)
                    t_hi = min(target_action[0,d]+self.h, 1)
                    lo = (t_lo - target_action[0,d])/self.h
                    hi = (t_hi - target_action[0,d])/self.h
                    den *= self.get_density(target_action[0,d],p[d]) * self.h * self.epanechnikov_int(lo, hi)
                val += num/den
                rewards.append(num/den)
            elif self.kernel == "boxcar":
                den = 1.
                num = l
                for d in range(target_action.shape[1]):
                    delta = (a[0][d] - target_action[0,d])/self.h
                    num *= self.boxcar_kernel(delta)
                    t_lo = max(target_action[0,d]-self.h, -1)
                    t_hi = min(target_action[0,d]+self.h, 1)
                    lo = (t_lo - target_action[0,d])/self.h
                    hi = (t_hi - target_action[0,d])/self.h
                    den *= self.get_density(target_action[0,d],p[d]) * self.boxcar_int(lo,hi) * self.h
                val += num/den
                rewards.append(num/den)
        #print("losses ", losses)
        return val/len(data)#, rewards
    
    def interval_overlap(self,sa_l, sa_h, b_l, b_h):
        '''
        Returns how much the interval (sa_l, sa_h) overlaps the interval (b_l, b_h)
        '''
        return max(min(sa_h,b_h)-max(sa_l, b_l), 0)

    def get_prob(self,a,p):
        if self.soften:
            prob = 1.
            for d in range(a.shape[1]):
                part1 = p[d]['rho_o'] * self.interval_overlap(a[0,d]-self.h, a[0,d]+self.h,0, p[d]['tau_low'])
                part2 = p[d]['rho_i'] * self.interval_overlap(a[0,d]-self.h, a[0,d]+self.h, p[d]['tau_low'], p[d]['tau_high'])
                part3 = p[d]['rho_o'] * self.interval_overlap(a[0,d]-self.h, a[0,d]+self.h, p[d]['tau_high'], 1)
                prob *= part1 + part2 + part3
            return prob
        else:
            """
            Get uniform density for the box of length h around a.
            This is complicated due to edge effects
            """
            total = 1
            for d in range(a.shape[1]):
                total *= min(a[0,d]+self.h,1)-max(a[0,d]-self.h,0)
            return total


    def variance(self,target,data):
        mean = 0
        zs = []
        for tup in data:
            (x,a,l,p) = tup
            target_action = target.get_action(x)['action']
            tmp = 0
            if self.kernel == None and np.all(np.abs(a - target_action) <= self.h):
                # boundary biasnot handled here, experiment at your own risk
                tmp = l/self.get_prob(target_action, p)
            elif self.kernel == "epanechnikov":
                den = 1.
                num = l
                for d in range(target_action.shape[1]): 
                    delta = (a[0][d] - target_action[0,d])/self.h
                    num *= self.epanechnikov_kernel(delta)
                    t_lo = max(target_action[0,d]-self.h, -1)
                    t_hi = min(target_action[0,d]+self.h, 1)
                    lo = (t_lo - target_action[0,d])/self.h
                    hi = (t_hi - target_action[0,d])/self.h
                    den *= self.get_density(target_action[0,d],p[d]) * self.h * self.epanechnikov_int(lo, hi)
                tmp += num/den
            elif self.kernel == "boxcar":
                den = 1.
                num = l
                for d in range(target_action.shape[1]):
                    delta = (a[0][d] - target_action[0,d])/self.h
                    num *= self.boxcar_kernel(delta)
                    t_lo = max(target_action[0,d]-self.h, -1)
                    t_hi = min(target_action[0,d]+self.h, 1)
                    lo = (t_lo - target_action[0,d])/self.h
                    hi = (t_hi - target_action[0,d])/self.h
                    den *= self.get_density(target_action[0,d],p[d]) * self.boxcar_int(lo,hi) * self.h
                tmp += num/den
            mean += tmp
            zs.append(tmp)
        mean = mean/len(data)
        return(np.mean([(z-mean)**2 for z in zs])/(len(zs)-1))
