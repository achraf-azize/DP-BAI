

import numpy as np
import math

import os
import uuid
import time

from bandit import Bandit

class AdaPTopTwo:
    def __init__(self, config):
        """
        Initializes the TopTwo algorithm with K arms.

        Args:
            K (int): The number of arms.
            beta (float): probability of playing the empirical best arm.
            eps (float): privacy parameter.
            delta (float): risk parameter.
        """
        #hyperparameters
        self.K = config["K"]
        self.eps = config["eps"]
        self.delta = config["delta"]
        self.beta = config["beta"]
        self.kappa = 1
        
        #name of the instance
        self.name = "OneDoublingTopTwo_id_"+uuid.uuid4().hex[:8] # add random id 
        if "name" in config.keys():
            self.name = config["name"]
            
        #name of the experiment
        self.exp_name = "FromTerminal"
        if "exp_name"in config.keys():
            self.exp_name = config["exp_name"]
                
        #total counts and mean-reward estimates
        self.counts = np.zeros(self.K)
        self.values = np.zeros(self.K)
        
        #phase statistics
        self.phase = np.zeros(self.K)
        self.ph_counts = np.zeros(self.K)
        self.ph_rewards = np.zeros(self.K)
        self.ph_best = None
        self.last_ph_counts = np.zeros(self.K)
        
        #useful to log info
        self.directory = "./experiments/" + self.exp_name +"/AdaPTT/"
        self.stopping = [] #store stats of stopping rule 
        self.info = dict()
        
    
    def run(self, bandit):
        """
        Runs private TopTwo algorithm on  bandit.

        Args:
            bandit (Bandit): instance of the Bandit class.
        """ 
        # Play each arm once to initialize
        for arm in range(self.K):
            reward = bandit.pull(arm)
            self.ph_counts[arm] += 1
            self.ph_rewards[arm] += reward
            self.update(arm)
        
        #main loop       
        while True : 
            arm = self.select_arm()
            reward = bandit.pull(arm)
            self.ph_counts[arm] += 1
            self.ph_rewards[arm] += reward
            if self.doubled_counts(arm):
                self.update(arm)
                #check if stopping rule is triggered
                if self.check_stopping():
                    # print(f"name : {self.name}, \n \
                    #     stopping time is {self.counts.sum()},\
                    #     best arm guess: {self.ph_best}")
                    self.save_logs()
                    return self.counts.sum(), np.argmax(self.values)
                
    
    def select_arm(self):
        """
        Selects which arm to play next using the sampling rule of TopTwo.

        Returns:
            int: The index of the arm to play.
        """
        #forced exploration
        # t = np.sum(self.ph_counts)
        # for arm in range(self.K):
        #     if self.ph_counts[arm] < np.sqrt(t):
        #         return arm
            
        # compute best arm & challenger from previous phase
        best = self.compute_priv_ucb_leader()
        challenger  = self.compute_challenger(best)
        
        if np.random.uniform() <= self.beta:
            return best
        else:
            return challenger

    def compute_priv_ucb_leader(self):
        """ 
        Computes the leader based on a private UCB index.
        """
        n = self.counts.sum()
        s = self.last_ph_counts
        ucb_index = self.values  +  np.log(n) / (self.eps*s)\
            + np.sqrt( np.log(n) / (2*s) )
        return np.argmax(ucb_index)
    
    def compute_challenger(self, best):
        """ 
        Computes the challenger arm based on estimates from previous phase.
        
        Args:
            best (int): index of the best arm
        """
        t = self.counts.sum()
        #stats of best arm
        n_best = self.counts[best] + self.ph_counts[best]
        v_best = self.values[best]

        challenger = None 
        minCost = np.inf
        for j in range(self.K):
            n_j = self.counts[j] + self.ph_counts[j]
            #transportation cost
            cost = (v_best - self.values[j]+ Kappa(t))\
                /np.sqrt(1/n_best +  1/n_j)
            if j != best and cost < minCost:
                challenger = j
                minCost = cost
        return challenger
    
    def check_stopping(self):
        """
        Checks whether the stopping condition is met.

        Args: 
            ph_counts (numpy.array): counts of the current phase 
    
        """
        #stats of best arm
        best = self.ph_best
        n_best = self.last_ph_counts[best]#+1e-10
        v_best = self.values[best]
        
        #check stopping rule
        for j in range(self.K):
            #transportation cost
            n_j = self.last_ph_counts[j]#+1e-10
            cost_j = 0.5*(v_best - self.values[j])**2 / (1/n_best +  1/n_j)
            threshold_j = emp_c(n_best, n_j, self.delta, self.eps)
            if j != best and  cost_j < threshold_j:
                return False
        return True
    
    def update(self, arm):
        """
        Updates phase number and total counts then constructs an eps-DP estimator of the mean rewards of every arm. 

        Args:
            arm (int): The index of the arm to update.
            reward (float): The reward received for playing the arm.
        """
        self.phase[arm] += 1
        self.counts[arm] += self.ph_counts[arm]
        self.last_ph_counts[arm] = self.ph_counts[arm]
        
        # forget previous rewards and construct new eps-DP estimate
        noise = np.random.laplace(scale = 1 / (self.eps* self.ph_counts[arm]))
        self.values[arm] = self.ph_rewards[arm]/self.ph_counts[arm] + noise
        
        #compute new best for next phase
        if (self.last_ph_counts == 0).any():
            self.ph_best = np.argmax(self.values)
        else:
            self.ph_best = self.compute_priv_ucb_leader()
 
        #reset phase stats
        self.ph_counts[arm] = 0
        self.ph_rewards[arm] = 0
        
    def doubled_counts(self, arm):
        """
        Checks whether the counts of arm have doubled.

        Args: 
            arm(int) : index of the arm
    
        """
        return (self.ph_counts[arm] == self.counts[arm]) 
    
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
        
        
            
           
def Kappa(t):
    """
    Computes the exploration bonus used in the challenger formula.
    
    Args: 
        t (int): total number of samples
    """
    alpha = 0.5
    return 1/(1+np.sqrt(t)) #np.log(1+t)** (-alpha/2)

def emp_c(n, m, delta, eps):
    term_0 = np.log(1/delta) + np.log(1 + np.log(n)) + np.log(1 + np.log(m))
    dp_term = (1/n + 1/m) * (1/eps**2) * np.log(1/delta)**2
    return term_0 + dp_term

def dp_c(n, m, K, ph, delta, eps):
    """
    Computes the private threshold of the stopping condition.

    Args: 
        n (int): counts of the current best arm
        m (int): counts of the challenger
        K (int): number of arms
        delta (float): risk parameter
        ph (int): index of the phase
        eps (float): privacy parameter

    """
    zeta = math.pi**2 / 6
    term_0 = 2*gaussian_calib(
        0.5 * np.log((K-1) * zeta * (ph**2) / delta)) \
        + 2*np.log(4+np.log(n)) + 2*np.log(4+np.log(m))
    dp_term = (1/n + 1/m) * (1/eps**2)\
        * np.log(2 * K * zeta * (ph**2) / delta)**2
    return term_0 + dp_term
    
def gaussian_calib(x):
    """
    Calibration function based on concentration of sum of KLs of gaussians.
    """
    return x+np.log(x)



if __name__=="__main__":
    K = 5
    mu = np.linspace(0, 1, 5)
    config = {"K": K, "beta": 0.5, "eps": 1.0, "delta": 0.1}
    for n in range(10):
        my_bandit = Bandit(K, mu)
        adap_top_two = AdaPTopTwo(config)
        adap_top_two.run(my_bandit)
