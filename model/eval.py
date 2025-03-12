import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import json

from architecture import TabFound

class OneDBenchmark:
    def __init__(self, choice, transform_input=True, device="cpu"):
        self.function = self._get_function(choice)
        self.transform_input = transform_input
        self.device = device

    def _get_function(self, choice):
        functions = {
            "branin": self.branin,
            "rosenbrock": self.rosenbrock,
            "ackley": self.ackley,
            "sine_wave": self.sine_wave,
        }
        if choice not in functions:
            raise ValueError(
                "Invalid choice. Please choose 'branin', 'rosenbrock', 'ackley', or 'sine_wave'."
            )
        return functions[choice]

    def _transform_input(self, x_norm, x_min, x_max):
        # Transforms x_norm in [0,1] to [x_min, x_max]
        return x_norm * (x_max - x_min) + x_min

    def _normalize_output(self, y, y_min, y_max):
        # Normalizes y (which should lie between y_min and y_max) to [0,1]
        return (y - y_min) / (y_max - y_min)

    def branin(self, x, **kwargs):
        if self.transform_input:
            x = self._transform_input(x, -5, 10)
        a = 1
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)
        y = 12.275
        out = a * (y - b * x**2 + c * x - r) ** 2 + s * (1 - t) * np.cos(x) + s
        return self._normalize_output(out, 0, 150)

    def rosenbrock(self, x, **kwargs):
        if self.transform_input:
            x = self._transform_input(x, -2, 2)
        a = 1
        out = (a - x) ** 2
        return self._normalize_output(out, 0, 10)

    def ackley(self, x, **kwargs):
        if self.transform_input:
            x = self._transform_input(x, -5, 5)
        a = 20
        b = 0.2
        c = 2 * np.pi
        out = -a * np.exp(-b * np.sqrt(x**2)) - np.exp(np.cos(c * x)) + a + np.exp(1)
        return self._normalize_output(out, 0, 15)

    def sine_wave(self, x, shift=0, **kwargs):
        if self.transform_input:
            # Transform x from [0,1] to [-10,10]
            x = self._transform_input(x, -10, 10)
        out = -torch.sin(x / 2 + math.pi / 2 + shift)
        return self._normalize_output(out, -1, 1)

    def __call__(self, x, **kwargs):
        result = self.function(x, **kwargs)
        return torch.tensor(result, dtype=torch.float32).to(self.device)


if __name__ == "__main__":

    steps = 20
    func = "branin"
    builder = OneDBenchmark(choice=func, transform_input=True, device="cpu")
    
    # Create a plot to visualize the periodic function.
    fig, ax = plt.subplots()
    # Generate 100 sample points for x in [0, 1]
    x_sample = np.linspace(0, 1, 100).astype(np.float32)
    # Convert to torch tensor with shape (100, 1)
    x_sample_tensor = torch.tensor(x_sample).unsqueeze(1)
    y_sample = builder(x_sample_tensor)
    ax.plot(x_sample, y_sample.cpu().numpy(), label=f"{func}")
    ax.legend()
    
    # Save the function plot in a designated directory
    output_dir = Path("/work/dlclarge1/janowski-opt/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    function_plot_path = output_dir / "function_plot.png"
    plt.savefig(str(function_plot_path))
    
    models_dir = "/work/dlclarge1/janowski-opt/models"
    # model_path = "output_250220"
    # model_path = "output_250221"
    date = "250226"
    # _id = "base"
    # _id = "traj_loss"
    _id = "traj_loss_100"
    model_path = f"output_{date}_{_id}"

    with open(f"{models_dir}/{model_path}/config.json", "r", encoding="utf-8") as f:
        config = json.load(f)


    # Load your neural network model architecture and weights
    nn_model = TabFound(
        input_features=config["input_features"],
        mean_embedding_value=config["mean_embedding_value"],
        nr_blocks=config["nr_blocks"],
        nr_heads=config["nr_heads"],
        dropout=config["dropout"],
        nr_hyperparameters=config["nr_hyperparameters"],
    )
    nn_model.load_state_dict(torch.load(f"{models_dir}/{model_path}/best_model.pt"))
    # Initialize with a starting x value in [0,1]; here we choose 0.5
    initial_x = np.array([[0.7]], dtype=np.float32)  # shape (1,1)
    initial_x = torch.tensor(initial_x)
    initial_y = builder(initial_x)  # Use our periodic function
    
    # Expand dimensions so that the data is shaped as (batch, sequence_length, features)
    # Here, we treat the two features as [x, y]
    # breakpoint()
    initial_x = initial_x.unsqueeze(0)  # shape (1,1,1)
    initial_y = initial_y.unsqueeze(0) # shape (1,1,1)
    x_values = torch.cat([initial_x, initial_y], dim=2)  # shape (1,1,2)
    
    # Now, run the model iteratively to generate a sequence of predictions.
    nn_model.eval()
    with torch.no_grad():
        for i in range(steps):
            y_pred = nn_model(x_values)
            # Use the first channel of the prediction as the new x value
            new_x = y_pred[0, i, 0]
            new_x = new_x.cpu().detach().numpy()
            new_x = np.array([[new_x]], dtype=np.float32)  # shape (1,1)
            new_x = torch.tensor(new_x)
            print(new_x)
            new_y = builder(new_x)  # Evaluate the periodic function on the new x
            # Prepare new entry with shape (1,1,2)
            new_x_exp = new_x.unsqueeze(0)
            new_y_exp = new_y.unsqueeze(0)
            new_entry = torch.cat([new_x_exp, new_y_exp], dim=2)
            # Append new entry to x_values along the sequence dimension
            x_values = torch.cat([x_values, new_entry], dim=1)
    
    # Plot the sequence predictions
    # breakpoint()
    indices = torch.arange(0, x_values.shape[1])
    print(indices)
    print(indices.shape)

    import matplotlib.colors as mcolors

    norma = mcolors.Normalize(vmin=indices.min().item(), vmax=indices.max().item())

    scatter = ax.scatter(
        x_values[0, :, 0].cpu().numpy(),
        x_values[0, :, 1].cpu().numpy(),
        c=indices.cpu().numpy(),
        cmap='viridis',
        label="Predictions",
        s=5,
        norm=norma,
    )
    plt.colorbar(scatter, ax=ax)

    # Save the final output plot to the results directory
    output_plot_path = output_dir / f"{func}_{model_path}.png"
    plt.savefig(str(output_plot_path))




# import math
# import os
# from pathlib import Path

# import numpy as np
# import pandas as pd
# import torch
# import matplotlib.pyplot as plt
# from generation.SineCosGenerator import SineGaussianBuilder
# from dill import load
# import os 
# from pathlib import Path
# import numpy as np


# if __name__ == "__main__":
#     #os.path.append(str(Path(__file__).parent))
#     with open("models_1.dill", 'rb') as file:
#         models = load(file)
#     model  = models[1]
#     builder  = SineGaussianBuilder(model[0], model[1], model[2], model[3], model[4])
#     fig, ax = plt.subplots()
#     builder.plot_fnc(ax)

#     model = TabFound(
#         input_features=2,
#         nr_blocks=6,
#         nr_heads=4,
#         dropout=0.2,
#         nr_hyperparameters=2,
#     )

#     model.load_state_dict(torch.load("best_model.pt"))


#     initial_x = [1]
#     initial_x = pd.DataFrame(initial_x)
#     #initial_y = obj(initial_x, fnc_list[0])
#     initial_y = builder.build_fnc(initial_x)
#     #print ("init y", initial_y.values)

#     initial_x = initial_x.values
#     initial_x = torch.tensor(initial_x, dtype=torch.float32)
#     initial_y = torch.tensor(initial_y.values, dtype=torch.float32)

#     #print ("initx", initial_x.shape)
#     #print ("inity", initial_y.shape)

#     initial_x = initial_x.unsqueeze(0)
#     initial_y = initial_y.unsqueeze(0)# .unsqueeze(0)

#     x_values = torch.cat([initial_x, initial_y], dim=2)

#     model.eval()
#     with torch.no_grad():
#         for i in range(100):
#             y_pred = model(x_values)
#             # print(x_values)
#             new_x = y_pred[0, i, 0]

#             # print(new_x)
#             new_x = new_x.cpu().detach().numpy()
#             new_x = np.expand_dims(new_x, axis=0)
#             new_x = pd.DataFrame(new_x)
#             #new_y = obj(new_x, fnc_list[0])
#             new_y = builder.build_fnc(new_x.values)
#             new_x = new_x.values
#             new_x = torch.tensor(new_x, dtype=torch.float32)
#             new_y = torch.tensor(new_y, dtype=torch.float32)
#             new_x = new_x.unsqueeze(0)
#             new_y = new_y.unsqueeze(0) # .unsqueeze(0)
#             new_x = torch.cat([new_x, new_y], dim=2)
#             x_values = torch.cat([x_values, new_x], dim=1)

#     # print(x_values)
#     indices = torch.arange(0, x_values.shape[1])
#     ax.scatter(x_values[0, :, 0], x_values[0, :, 1], c=indices, cmap='Blues', label="Predictions", s=5)
#     plt.savefig("output.png")
#     #plt.show()