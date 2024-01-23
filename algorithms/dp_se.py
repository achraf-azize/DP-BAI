

import numpy as np
import os
import uuid
from bandit import Bandit

class DPSE:
    def __init__(self, config):
        """
        Initializes the DP-SE algorithm with K arms.

        Args:
            K (int): The number of arms.
            self.delta (float): probability of playing the empirical best arm.
            eps (float): privacy parameter.
            delta (float): risk parameter.
        """
        #hyperparameters
        self.K = config["K"]
        self.eps = config["eps"]
        self.delta = config["delta"]
        
        #name of the instance
        self.name = "DPSE_id_"+uuid.uuid4().hex[:8] # add random id 
        if "name" in config.keys():
            self.name = config["name"]
            
        #name of the experiment
        self.exp_name = "FromTerminal"
        if "exp_name"in config.keys():
            self.exp_name = config["exp_name"]
                
        #total counts and mean-reward estimates
        self.counts = np.zeros(self.K)
        self.values = np.zeros(self.K)
        self.dp_values = np.zeros(self.K)
        
        #others
        self.epoch = 0
        self.active_set = list(np.arange(self.K))
        
        #logging info
        self.directory = "./experiments/" + self.exp_name +"/DPSE/"
        self.stopping = [] #store stats of stopping rule 
        self.info = dict()
        
    def run(self, bandit):
        """
        Runs private DP-SE algorithm on bandit.

        Args:
            bandit (Bandit): instance of the Bandit class.
        """   
        while len(self.active_set) > 1 : 
            # print(f"name : {self.name}, \n \
            #           t == {self.counts.sum()}") 
            self.epoch += 1
            r = 0
            self.reset_values()
            Re = self.compute_Re()
            
            for arm in self.active_set:
                rewards = []
                for k in range(int(Re)):
                    reward = bandit.pull(arm)
                    rewards.append(reward)
                self.update(arm, rewards)

            self.privatize(int(Re)) 
            self.do_eliminations(Re)      
        
        # print(f"name : {self.name}, \n \
        #                 stopping time is {self.counts.sum()},\
        #                 best arm guess: {np.argmax(self.dp_values)}")
        self.save_logs()
        return self.counts.sum(), np.argmax(self.dp_values)


    def do_eliminations(self, Re):
        #Eliminates sub-optimal arms
        h_e = np.sqrt( np.log(8 * len(self.active_set) * (self.epoch**2)\
            / self.delta) / (2*Re) )
        c_e = np.log(4 * len(self.active_set) * (self.epoch**2) / self.delta )  / (Re*self.eps) 
        
        S = self.active_set.copy() 
        max_mu  = np.max(self.dp_values)
        for arm in S:
            if max_mu - self.dp_values[arm] > 2*(h_e + c_e):
                self.active_set.remove(arm)
            
    
    def privatize(self, r):
        scale = 1/ (self.eps * r)
        self.dp_values = self.values + np.random.laplace(scale = scale, size = self.K)
        
    def update(self, arm, rewards):
        n = self.counts[arm]
        k = len(rewards)
        self.values[arm] = (n*self.values[arm] + sum(rewards))/ (n+k)
        self.counts[arm] += k
    
    def compute_Re(self):
        #compute epoch parameter Re
        Delta_e = 2**(-self.epoch)
        
        Re1 = 32*np.log(8 * len(self.active_set) * (self.epoch**2)\
            / self.delta) / (Delta_e**2)
        Re2 = 8*np.log(4 * len(self.active_set) * (self.epoch**2)\
            / self.delta) /  (Delta_e * self.eps)
        
        Re = max(Re1, Re2) + 1
        return Re
    
    def reset_values(self):
        self.values = np.zeros(self.K)
        
    def save_logs(self):
        #save logs of experiments in dedicated file
        tau  = self.counts.sum()
        a_star = np.argmax(self.values)

        if not os.path.isdir(self.directory):
            try:
                os.makedirs(self.directory)
            except FileExistsError:
                print(f"{self.name} file exists")
        filename = self.directory+self.name+".csv"
        
        data = np.array([self.K, self.delta, self.eps, tau, a_star])
        np.savetxt(filename, data, delimiter=",")
        
        
if __name__=="__main__":
    K = 5
    mu = np.linspace(0, 1, 5)
    my_bandit = Bandit(K, mu)
    
    config = {"K": K,  "eps": 1.0, "delta": 0.1}
    dp_se = DPSE(config)
    astar, tau = dp_se.run(my_bandit)
    print(astar, tau)