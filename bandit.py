import numpy as np

class Bandit:
    def __init__(self, mu = None, K= None):
        self.mu = mu
        self.K = len(mu)  # number of arms
        if self.mu is None:
            if self.K is None:
                raise ValueError("Both mu and K are not specified")
            self.mu = np.random.uniform(size=K)  # success probabilities of each arm
        
    def pull(self, arm):
        return np.random.binomial(1, self.mu[arm])
