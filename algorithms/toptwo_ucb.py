
import numpy as np
import math
import os
import uuid
from bandit import Bandit

class TTUCB:
    def __init__(self, config):
        """
        Initializes the TopTwo-UCB algorithm with K arms.

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
        if "kappa" in config.keys():
            self.Kappa = config["kappa"]
        else:
            self.Kappa = Kappa
        
        #name of the instance
        self.name = "VanillaTopTwo_id_"+uuid.uuid4().hex[:8] # add random id 
        if "name" in config.keys():
            self.name = config["name"]
            
        #name of the experiment
        self.exp_name = "FromTerminal"
        if "exp_name"in config.keys():
            self.exp_name = config["exp_name"]
                
        #total counts, cumulated rewards and mean-reward estimates
        self.counts = np.zeros(self.K)
        self.rewards = np.zeros(self.K)
        self.values = np.zeros(self.K)
        
        #current estimate of best arm
        self.best = None
        
        #useful to log info
        self.directory = "./experiments/" + self.exp_name +"/TTUCB/"
        self.stopping = [] #store stats of stopping rule 
        self.info = dict()
        
    
    def run(self, bandit):
        """
        Runs vanilla TopTwo algorithm on  bandit.

        Args:
            bandit (Bandit): instance of the Bandit class.
        """
        #intialization phase
        for arm in range(self.K):
            reward = bandit.pull(arm)
            self.update(arm, [reward])
        
        #main loop       
        while True : 
            arm = self.select_arm()
            reward = bandit.pull(arm)
            self.update(arm, [reward]) #update counts and values  
            #check if stopping rule is triggered
            if self.check_stopping():
                # print(f"name : {self.name}, \n \
                #     stopping time is {self.counts.sum()},\
                #         best arm guess: {self.best}")
                self.save_logs()
                return self.counts.sum(), np.argmax(self.values)
                
    
    def select_arm(self):
        """
        Selects which arm to play next using the sampling rule of TopTwo.

        Returns:
            int: The index of the arm to play.
        """
        # compute challenger e
        best = self.best
        challenger  = self.compute_challenger(best)
        
        if np.random.uniform() <= self.beta:
            return best
        else:
            return challenger

    def compute_challenger(self, best):
        """ 
        Computes the challenger arm based on estimates from previous phase.
        
        Args:
            best (int): index of the best arm
        """
        # stats of best arm
        n_best = self.counts[best] 
        v_best = self.values[best]
        
        # computing the exploration parameter  
        t = self.counts.sum() #number of samples by the algo
        kappa = self.Kappa(t)
        
        #compute challenger
        challenger = None 
        minCost = np.inf
        for j in range(self.K):
            n_j = self.counts[j]
            #transportation cost
            cost = (v_best - self.values[j]+ kappa)\
                /np.sqrt(1/n_best +  1/n_j)
            if j != best and cost < minCost:
                challenger = j
                minCost = cost
        return challenger
    
    def check_stopping(self):
        """
        Checks whether the stopping condition is verified.
    
        """
        t = self.counts.sum()
        
        #stats of best
        best = self.best
        n_best = self.counts[best] 
        v_best = self.values[best]
        
        for j in range(self.K):
            #transportation cost
            n_j = self.counts[j] #counts of the previous phase of j
            cost_j = 0.5*(v_best - self.values[j])**2 / (1/n_best +  1/n_j)
            threshold_j = emp_c(t, self.delta)
            
            if j != best and cost_j < threshold_j:
                # print(f"step {self.counts.sum()} \
                #     arm {j}, ratio = {cost_j / threshold_j}, \
                #         counts = {self.counts}, values = {self.values}")
                return False  
        return True
    
    def update(self, arm, rewards):
        """
        Updates count and mean-reward of pulled arm, computes new best arm  

        Args:
            arm (int): The index of the arm to update.
            rewards (numpy.array): array of rewards received when playing the arm.
        """
        m = len(rewards)
        n = self.counts[arm]
        self.values[arm] = (n*self.values[arm] + m*np.sum(rewards))/ (n+m)
        self.counts[arm] += m
        
        #compute new best arm
        self.best = np.argmax(self.values)
    
        
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
        t (int): total number of samples used so far.
    """
    alpha = 0.5
    return np.log(1+t)** (-alpha/2)

def emp_c(t, delta):
    return np.log( (1+np.log(t)) / delta)

def c(n, m, K, t, delta):
    """
    Computes the non-private threshold of the stopping condition.

    Args: 
        n (int): counts of the current best arm
        m (int): counts of the challenger
        K (int): number of arms
        delta (float): risk parameter
        t (int): total number of samples

    """
    zeta = math.pi**2/6
    return 2*gaussian_calib(
        0.5 * np.log((K-1) * zeta * (t**2) / delta)) \
        + 2*np.log(4+np.log(n)) + 2*np.log(4+np.log(m))

def gaussian_calib(x):
    """
    Calibration function based on concentration of sum of KLs of gaussians.
    """
    return x+np.log(x)



if __name__=="__main__":
    K = 5
    mu = np.linspace(0, 1, 5)
    my_bandit = Bandit(K, mu)
    config = {"K": K, "beta": 0.5, "eps": 1.0, "delta": 0.1}
    vanilla_TT = TTUCB(config)
    vanilla_TT.run(my_bandit)
