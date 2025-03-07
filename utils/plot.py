import pandas as pd
import matplotlib.pyplot as plt
from generation.SimpleTrig_1D import rec_fnc, obj, eval_x
from dill import load
import os 
from pathlib import Path
import numpy as np
from tqdm import tqdm

def load_batch(filename):
    f = np.load(filename)
    return f["x"], f["y"]

if __name__=="__main__":
    #os.path.append(str(Path(__file__).parent))
    with open("./datasets/single/1D_triv/models_0.dill", 'rb') as file:
        models = load(file)

    x_hebo, y_hebo = load_batch("./datasets/single/1D_triv/data_0.npz")
    print (x_hebo.shape, y_hebo.shape)
    n_fncs = 5
    hebo_trials = 4
    
    fig, axes = plt.subplots(n_fncs, hebo_trials, figsize=(20, 30))
    for model_idx in tqdm(range(n_fncs)):
        
        gt  = models[model_idx]
        x_rec, y_rec = rec_fnc(gt)
        #fig, ax = plt.subplots()
        for n_trial in range(hebo_trials):
            print ((model_idx*hebo_trials)+ (n_trial))
            ax = axes[model_idx, n_trial]
            ax.plot(x_rec, y_rec, c="b") #, s=1)
            ax.set_ylim(0,1)
            ax.scatter(x_hebo[(model_idx*hebo_trials)+ (n_trial)], y_hebo[model_idx*hebo_trials + n_trial], 
                       c=range(x_hebo.shape[1]), cmap="Reds", marker="X", s=50)
        #ax = axes[model_idx]
        #x_rec, y_rec = rec_fnc(gt)
        #ax.scatter(x_rec, y_rec, c="black", s=3)
        #ax.scatter(x_hebo[model_idx], y_hebo[model_idx], cmap="Reds", s=3)
    fn = "test.png"
    plt.savefig(fn)
    plt.tight_layout()
    print(f"Plot saved as {fn}")
