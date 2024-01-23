import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import itertools

from multiprocessing import Pool
from bandit import Bandit

from algorithms.toptwo_ucb import TTUCB
from algorithms.adapTT import AdaPTopTwo
from algorithms.dp_se import DPSE

from utils import read_results




def run_simulation(config): 
    #instantiate bandit 
    mu = config["mu"]
    bandit = Bandit(mu)
    
    #instantiate algorithm
    algos = {"tt_ucb": TTUCB, "adap_tt": AdaPTopTwo, "dp_se": DPSE }
    algo_name = config["algo_name"] 
    algorithm = algos[algo_name](config)
    
    #run algorithm on bandit instance
    algorithm.run(bandit)

if __name__ == "__main__":
    #Experiment name
    exp_id = "linspace_10simulations_4workers"
    exp_name = f"Experiment_{exp_id}"
    
    #hyperparameters
    Deltas = [1e-2]
    Epsilons = [0.1, 1.0] #10**(np.linspace(-5, 5,11))
    combinations = list(itertools.product(Epsilons, Deltas))
    num_simulations = 10 #100
    num_workers = 4 #32

    #means of the bandit instance
    mu = np.linspace(0, 1, 5)
    #mu = np.array([0.5, 0.9, 0.9, 0.9, 0.95])
    #mu = np.array([0.75,0.70,0.70,0.70,0.70])
    #mu = np.array([0.75,0.625,0.5,0.375,0.25])
    #mu = np.array([0.75,0.53125,0.375,0.28125,0.25])
    #mu = np.array([0.75,0.71875,0.625,0.46875,0.25])
    K = len(mu)
    
    for (epsilon, delta) in combinations:
        for alg in ["tt_ucb", "adap_tt","dp_se"]:
            config = {"exp_name": exp_name, "mu": mu, "K": K, "algo_name": alg,\
                "beta": 0.5,"eps": epsilon, "delta": delta}
            
            pool = Pool(processes = num_workers) # create num_workers processes
            results = pool.map(run_simulation, [(config)]*num_simulations)
            

    
    subfolders = ["/TTUCB", "/AdaPTT","/DPSE" ] 
    data = []
    names = []
    fig, ax = plt.subplots()
    for sub in subfolders:
        folder = "./experiments/" + exp_name + sub
        name, epsilons, taus, astars = read_results(folder)
        X = epsilons
        #taus = np.log(taus)
        Y_mean = np.mean(taus, axis = 1)
        Y_std = np.std(taus, axis = 1)
        ax.plot(X, Y_mean, label=name)
        ax.fill_between(X, Y_mean - Y_std, Y_mean + Y_std,
                        alpha=0.2)
    
    #Set the y-axis label to scientific notation
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax.yaxis.offsetText.set_visible(True)
    #ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
    ax.legend(loc='best')
    ax.set_xlabel('epsilons')
    ax.set_title('log-sample complexity')
    plt.savefig(f"./figures/experiment_{exp_id}", format = "pdf")
    plt.show()
    
