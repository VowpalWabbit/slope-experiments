import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import Params

torch.set_default_tensor_type(torch.DoubleTensor)

class NNPredictor(object):
    class NNModel(nn.Module):
        def __init__(self,input_dim,output_dim):
            super(NNPredictor.NNModel,self).__init__()
            self.network = nn.Sequential(nn.Linear(input_dim,output_dim,bias=False),
                                         nn.Sigmoid())
        def forward(self,x):
            return (self.network(x))

    def __init__(self, input_dim, output_dim):
        self.model = NNPredictor.NNModel(input_dim, output_dim)
        self.criterion = nn.MSELoss()

    def fit(self, X, y):
        optimizer = optim.Adagrad(self.model.parameters(), lr = 0.1)
        prev_loss = 0.0
        for i in range(5000):
            total_loss = 0.0
            for a in range(y.shape[1]):
                optimizer.zero_grad()
                preds = self.model(torch.tensor(X))
                loss = self.criterion(preds[:,a], torch.tensor(y[:,a]))
                total_loss += loss
                loss.backward()
                optimizer.step()
            if np.mod(i,100) == 0:
                x = total_loss.detach().numpy()
                if np.round(x,3) == np.round(prev_loss,3):
                    break
                prev_loss = x

    def predict(self, x):
        return self.model(torch.tensor(x)).detach().numpy()



class CCBPolicy(object):
    def __init__(self,act_dim,model=None):
        self.model=model
        self.act_dim=act_dim
    
    
    def friendly_soften(self, action, soften_params):
        alpha = soften_params['alpha']
        beta = soften_params['beta']
        n_bins = soften_params['l']
        soft_action = []
        probs = []
        for act in action[0]:
            bin_id = np.ceil(act * n_bins)
            tau_low = (bin_id-1.)/n_bins
            tau_high = bin_id*1./n_bins

            u = np.random.uniform(-0.5, 0.5)
            explore_prob = alpha+beta*u
            if np.random.uniform(0,1) > explore_prob:
                bin_id = np.random.choice(list(set(np.arange(1,n_bins+1))-set([bin_id])))
            chosen_act = np.random.uniform((bin_id-1.)/n_bins,bin_id*1./n_bins)
            soft_action.append(chosen_act)
            
            rho_o = ((1 - explore_prob) * (n_bins/(n_bins - 1)))
            rho_i = (explore_prob * n_bins)
            
            prob_data = {'tau_low': tau_low, 'tau_high': tau_high, 'rho_o': rho_o, 'rho_i': rho_i}
            probs.append(prob_data)
            
        return soft_action, probs

    def adversarial_soften(self, action, soften_params):
        alpha = soften_params['alpha']
        beta = soften_params['beta']
        n_bins = soften_params['l']
        soft_action = []
        probs = []
        for act in action[0]:
            """
            1. tau_low and tau_high are the lower and upper bounds of the bin in 
            which the unsoftened action falls.
            2. rho_o is the density outside (tau_low, tau_high) after softening
            and rho_i is the density inside (tau_low, tau_high) after softening
            """

            bin_id = np.ceil(act * n_bins)
            tau_low = (bin_id-1.)/n_bins
            tau_high = bin_id*1./n_bins
            
            u = np.random.uniform(-0.5, 0.5)
            explore_prob = alpha+beta*u
            if np.random.uniform(0,1) < alpha+beta*u:
                bin_id = np.random.choice(list(set(np.arange(1,n_bins+1))-set([bin_id])))
            else:
                bin_id = np.random.choice(list(set(np.arange(1,n_bins+1))))
            chosen_act = np.random.uniform((bin_id-1.)/n_bins, bin_id*1./n_bins)
            soft_action.append(chosen_act)
            
            rho_o = (explore_prob * (n_bins/(n_bins - 1))) + (1-explore_prob)
            rho_i = (1 - explore_prob)
            
            prob_data = {'tau_low': tau_low, 'tau_high': tau_high, 'rho_o': rho_o, 'rho_i': rho_i}
            probs.append(prob_data)
            
        return soft_action, probs
    
    def neutral_soften(self, action, soften_params):
        soft_action = np.random.uniform(0,1,[1,self.act_dim])[0]
        probs = []
        for i in range(self.act_dim):
            prob_data = {'tau_low': 0, 'tau_high': 1, 'rho_o': 0, 'rho_i': 1}
            probs.append(prob_data)
        return soft_action, probs
    
    def get_soften_action(self, action, soften_params):
        if soften_params != None:
            soft_action = []
            if soften_params['method'] == "friendly":
                soft_action, probs = self.friendly_soften(action, soften_params)
            elif soften_params['method'] == "adversarial":
                soft_action, probs = self.adversarial_soften(action, soften_params)
            elif soften_params['method'] == "neutral":
                soft_action, probs = self.neutral_soften(action, soften_params)
            return {"action": [soft_action], "prob": probs}
        else:
            probs = []
            for i in range(self.act_dim):
                prob_data = {'tau_low': 0, 'tau_high': 1, 'rho_o': 0, 'rho_i': 1}
                probs.append(prob_data)
            return {"action": action, "prob": probs}

    def get_action(self, x, soften=None):
        
        if soften == "friendly":
            soften_params = Params.FRIENDLY
        elif soften == "adversarial":
            soften_params = Params.ADVERSARIAL
        elif soften == "neutral":
            soften_params = Params.NEUTRAL
        else:
            soften_params = None
            
        if self.model is None:
            act = np.random.uniform(0,1,[1,self.act_dim])
        else:
            if self.act_dim == 1:
                act = self.model.predict(x).reshape(-1,1)
            else:
                act = self.model.predict(x)
            act = torch.clamp(torch.Tensor(act),0,1).detach().numpy()
        act_prob = self.get_soften_action(act, soften_params)
        return act_prob

class CCBSimulatedEnv(object):
    def __init__(self, lip=1, feat_dim=5, act_dim = 1, target_model_name = "NNPredictor", logging_model_name = None, loss_type="triangular", soften=None):
        self.feat_dim=feat_dim
        self.act_dim=act_dim
        self.logging_model = self.get_model(logging_model_name)
        self.target_model = self.get_model(target_model_name)
        self.lip=lip
        self.opt=np.random.normal(0,1,(self.feat_dim,self.act_dim))
        self.loss_type = loss_type
        self.soften=soften

    def get_model(self, model_name):
        if model_name == "NNPredictor":
            return NNPredictor(self.feat_dim,self.act_dim)
        elif model_name == "Tree":
            #return DecisionTreeRegressor(max_depth=5, min_samples_split = 5, min_samples_leaf = 5)
            return RandomForestRegressor(random_state=1, n_estimators=10, min_samples_split=5)
        else:
            return None
        
    def train_logger(self, n=0, sig=0.5):
        """
        Uniform logging policy for now
        """
        if self.soften:
            if self.logging_model != None:
                X = np.zeros((n, self.feat_dim))
                Y = np.zeros((n, self.act_dim))
                for i in range(n):
                    x = self.context()
                    X[i,:] = x
                    Y[i,:]= self.get_center(x) + np.random.normal(0,sig,(1,self.act_dim))
        
                self.logging_model.fit(X,Y)
            else:
                raise ValueError("soften without model?!! can't do!!!")
            self.logger = CCBPolicy(self.act_dim, model = self.logging_model)
        else:
            self.logger = CCBPolicy(self.act_dim)

    def train_target(self, n,sig=0.5):
        """
        Good target policy trained via logistic regression (effectively)
        """
        X = np.zeros((n, self.feat_dim))
        Y = np.zeros((n, self.act_dim))
        for i in range(n):
            x = self.context()
            X[i,:] = x
            Y[i,:]= self.get_center(x) + np.random.normal(0,sig,(1,self.act_dim))

        self.target_model.fit(X,Y)
        self.target = CCBPolicy(self.act_dim,model=self.target_model)

    def ground_truth(self,n):
        '''
        if self.soften == "friendly":
            soften_params = Params.FRIENDLY
        elif self.soften == "adversarial":
            soften_params = Params.ADVERSARIAL
        elif self.soften == "neutral":
            soften_params = Params.NEUTRAL
        else:
            soften_params = None'''
        soften_params = None
        l = 0
        for i in range(n):
            x = self.context()
            a = self.target.get_action(x,soften_params)['action']
            l += self.loss(x,a)
        return(l/n)

    def get_center(self,x):
        return np.exp(np.dot(x,self.opt))/(1+np.exp(np.dot(x,self.opt)))

    def loss(self, x, a):
        center = self.get_center(x)
        if self.loss_type == "triangular":
            loss = min(np.sum(self.lip*np.abs(a - center)),1)
        elif self.loss_type == "parabolic":
            loss = min(np.sum((self.lip**2/4)*np.abs(a-center)),1)
        else:
            loss = 1.
        return loss

    def context(self):
        x = np.random.normal(0,1,[1,self.feat_dim])
        return(x)

    def gen_logging_data(self,n):
        """
        @akshay: only supports uniform logging for now
        """
        data = []
        for i in range(n):
            x = self.context()
            act_prob = self.logger.get_action(x, self.soften)
            a = act_prob["action"]
            p = act_prob["prob"]
            #print("prob: ", p)
            l = self.loss(x,a)
            data.append((x,a,l,p))
        return (data)

if __name__=='__main__':
    
    Env = CCBSimulatedEnv(lip=3,act_dim=2)
    Env.train_logger()
    Env.train_target(100)
    print("Ground truth loss: %0.2f" % (Env.ground_truth(1000)))

    data = Env.gen_logging_data(10000)
    print("Uniform exploration average loss: %0.2f" % (np.mean([tup[2] for tup in data])))
   
    """
    Env = CCBSimulatedEnv(lip=3, act_dim=2, target_model_name = "NNPredictor", logging_model_name = "NNPredictor")
    Env.train_logger(10000)
    Env.train_target(3000)
    soften_target_params = None
    soften_logging_params = {'method': 'friendly', 'alpha': 0.7, 'beta': 0.2, 'l': 10}
    print("Ground truth loss: %0.2f" % (Env.ground_truth(100000, soften_target_params)))
    data = Env.gen_logging_data(1000, soften_logging_params)
    print("Uniform exploration average loss: %0.2f" % (np.mean([tup[2] for tup in data])))
    """

    #import SmoothEval
    #print("Off policy estimate: %0.2f" % (SmoothEval.smooth_eval(Env.target, data, 0.1)))
