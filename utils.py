import numpy as np
import os
import glob
### Useful functions for processing and plots


def read_results(folder):
    """
    Reads .txt files from experiment folder and returns data
    """
    #algorithm name
    name = folder.split("/")[3]
    
    #read files and store data
    epsilons, taus, astars = [], [], []
    index = dict()
    last_index = 0
    for file_path in glob.glob(os.path.join(folder, "*")):
        if os.path.isfile(file_path):
            with open(file_path, "rb") as file:
                data = np.loadtxt(file)
                epsilon = data[2]
                #if value of epsilon not encountered yet
                if not (str(epsilon) in index.keys()):
                    epsilons.append(epsilon)
                    #set index of epsilon
                    index[str(epsilon)] = last_index
                    last_index +=1
                    #create empty lists where its corresponding data will be stored
                    taus.append([])
                    astars.append([])
                    
                idx = index[str(epsilon)]
                taus[idx].append(data[3])
                astars[idx].append(data[4]) 
    
    #sort epsilons and their correspoding taus for later plots
    epsilons, taus, astars = sort(epsilons, taus, astars, index)       
    return name, epsilons, np.array(taus), np.array(astars)

def sort(epsilons, taus, astars, index):
    """ 
    Sort epsilons in increasing orders
    Args:
        epsilons (list): list of (unordered) epsilon parameters used for the DP-BAI experiments
        taus (list): list of stopping times. Has the same order of epsilons: taus[i] corresponds to the stopping time of epsilons[i]
        astars (list): list of best arm guesses. Has the same order of epsilons. 
        index (dict): dictionary containing the indices of each value of epsilon within epsilons
    """
    #
    epsilons = np.sort(np.array(epsilons)) 
    sorted_taus = taus.copy()
    sorted_astars = astars.copy()
    for i, epsilon in enumerate(epsilons):
        idx = index[str(epsilon)]
        sorted_taus[i] = taus[idx]
        sorted_astars[i] = astars[idx]
    
    return epsilons, sorted_taus, sorted_astars