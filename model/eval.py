from .architecture import TabFound
import pandas as pd
import torch
import matplotlib.pyplot as plt
from generation.SineCosGenerator import SineGaussianBuilder
from generation.SimpleTrig_1D import rec_fnc, obj, eval_x
from dill import load
import os 
from pathlib import Path
import numpy as np
from tqdm import tqdm


if __name__ == "__main__":
    #os.path.append(str(Path(__file__).parent))
    with open("models_0.dill", 'rb') as file:
        models = load(file)

    model_idx = range(5)

    fig, axes = plt.subplots(5,1, figsize=(10, 20))
    for model_idx in tqdm(range(5)):
        
        gt  = models[model_idx]
        #fig, ax = plt.subplots()
        
        ax = axes[model_idx]
        x_rec, y_rec = rec_fnc(gt)
        ax.scatter(x_rec, y_rec, c="black", s=3)

        #builder  = SineGaussianBuilder(model[0], model[1], model[2], model[3], model[4])
        #builder.plot_fnc(ax)

        model = TabFound(
            input_features=2,
            mean_embedding_value=64,
            nr_blocks=8,
            nr_heads=4,
            dropout=0.2,
            nr_hyperparameters=2,
        )

        model.load_state_dict(torch.load("output_250227/trial_1/best_model.pt"))


        initial_x = [1]
        initial_x = pd.DataFrame(initial_x)
        #initial_y = obj(initial_x, fnc_list[0])
        # initial_y = builder.build_fnc(initial_x)
        initial_y = eval_x(initial_x, gt)
        #print ("init y", initial_y.values)

        initial_x = initial_x.values
        initial_x = torch.tensor(initial_x, dtype=torch.float32)
        initial_y = torch.tensor(initial_y.values, dtype=torch.float32)

        #print ("initx", initial_x.shape)
        #print ("inity", initial_y.shape)

        initial_x = initial_x.unsqueeze(0)
        initial_y = initial_y.unsqueeze(0) # .unsqueeze(0)

        x_values = torch.cat([initial_x, initial_y], dim=2)

        model.eval()
        with torch.no_grad():
            for i in tqdm(range(25)):
                y_pred = model(x_values)
                new_x = y_pred[0, i, 0]

                new_x = new_x.cpu().detach().numpy()
                new_x = np.expand_dims(new_x, axis=0)
                new_x = pd.DataFrame(new_x)
                new_y = eval_x(new_x.values, gt)
                # new_y = builder.build_fnc(new_x.values)
                new_x = new_x.values
                new_x = torch.tensor(new_x, dtype=torch.float32)
                new_y = torch.tensor(new_y, dtype=torch.float32)
                new_x = new_x.unsqueeze(0)
                new_y = new_y.unsqueeze(0) # .unsqueeze(0)
                new_x = torch.cat([new_x, new_y], dim=2)
                x_values = torch.cat([x_values, new_x], dim=1)

        indices = torch.arange(0, x_values.shape[1])
        ax.scatter(x_values[0, :, 0], x_values[0, :, 1], c=indices, cmap="Reds", label="Predictions", s=50)
    plt.savefig("output.png")
    plt.tight_layout()
    print("Plot saved as output.png")
