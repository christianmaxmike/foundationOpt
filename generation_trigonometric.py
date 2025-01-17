from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from tqdm import tqdm
import pandas as pd
import numpy as np
from dill import dump, load
import argparse
np.random.seed(42)


def chris_fnc_random(x, min_num_components=1, max_num_components=8) -> np.ndarray:
    batch_y = np.zeros((x.shape[0], x.shape[1]))
    for batch_idx, batch in enumerate(range(x.shape[0])):
        num_components = np.random.randint(min_num_components, max_num_components)
        y_values = np.zeros((x.shape[1]), dtype=np.float16)
        for _ in range(num_components):
            amplitude = np.random.uniform(0.5, 2.0) 
            y_tmp = np.ones((x.shape[1]), dtype=np.float16)
            for dim_idx , dim in enumerate(range(x.shape[1])):
                trigonometric = np.sin if np.random.rand() < 0.5 else np.cos
                frequency = np.random.uniform(0.05, 0.25) 
                phase = np.random.uniform(1,5) 
                y_tmp *= trigonometric(frequency * x[batch_idx, :, dim_idx] + phase)
            y_values += amplitude * y_tmp
        batch_y[batch_idx] = y_values
    return batch_y


def obj(params : pd.DataFrame, fnc_dict) -> np.ndarray:
    """
    Load function from the attached function dictionary and return y values of the x values stored in the attached
    parameter (params) dataframe.
    """
    #print(params.iloc[:, 0].values)

    y_values = np.zeros_like(params.iloc[:, 0].values, dtype=np.float16)
    for i in range(fnc_dict['num_components']):
        #print(f"Processing component number {i}")
        y_tmp = np.ones_like(params.iloc[:, 0].values, dtype=np.float16)
        amplitude = fnc_dict['amplitudes'][i]  # Amplitude between 0.5 and 2.0

        for dim_idx , dim in enumerate(range(len(fnc_dict['frequencies'][0]))):
          frequency = fnc_dict['frequencies'][i][dim_idx]
          phase = fnc_dict['phases'][i][dim_idx]
          trigonometric = fnc_dict['trigonometrics'][i][dim_idx]
          y_tmp *= trigonometric(frequency * params.iloc[:, dim_idx].values + phase)
        y_values += amplitude * y_tmp

    # normalize values
    y_values = (y_values - np.min(y_values)) / (np.max(y_values) - np.min(y_values))
    return y_values


def gen_trigonometric_fncs(batch_size, dims, min_x_value, max_x_value, sequence_length, min_num_components, max_num_components):
    print(f"Start generating fnc with dimension {dims} of length {sequence_length}.")    
    # list containing synthesized trigonometric functions
    fnc_list = []

    # generation loop
    for i in range(batch_size):
        fnc_dict = {}
        amplitudes = []
        frequencies = []
        phases = []
        trigonometrics = []

        # Randomize parameters for multiple components
        num_components = np.random.randint(min_num_components, max_num_components)  # Number of sine/cosine components

        for _ in range(num_components):
            amplitude = np.random.uniform(0.05, 1.0)  # Amplitude between 0.5 and 2.0

            comp_freq = []
            comp_phases = []
            comp_trigonometrics = []

            for dim_idx , dim in enumerate(range(dims)):
                trigonometric = np.sin if np.random.rand() < 0.5 else np.cos
                frequency = np.random.uniform(1, 15) #(0.5, 2.0)  # Frequency between 0.5 and 2.0
                phase = np.random.uniform(1,10) #(0, 2 * np.pi)  # Phase shift

                comp_freq.append(frequency)
                comp_phases.append(phase)
                comp_trigonometrics.append(trigonometric)

            frequencies.append(comp_freq)
            phases.append(comp_phases)
            trigonometrics.append(comp_trigonometrics)
            amplitudes.append(amplitude)

        # add values to function dictionary
        fnc_dict['amplitudes'] = amplitudes
        fnc_dict['frequencies'] = frequencies
        fnc_dict['phases'] = phases
        fnc_dict['trigonometrics'] = trigonometrics
        fnc_dict['num_components'] = num_components

        fnc_list.append(fnc_dict)
    return fnc_list


def run_hebo(num_hebo_steps, fnc_list, dims, x_lb, x_ub):
    # HEBO loop
    hebo_xs = []
    hebo_ys = []
    for fnc_idx in range(len(fnc_list)):
        print (f"Processing fnc {fnc_idx}...")
        space = DesignSpace().parse([{'name' : f'x{idx}', 'type' : 'num', 'lb' : x_lb, 'ub' : x_ub} for idx in range(dims)])
        opt   = HEBO(space)
        x_hebo = []
        y_hebo = []
        for i in tqdm(range(num_hebo_steps)):
            rec = opt.suggest(n_suggestions = 1)
            y_value = obj(rec, fnc_list[fnc_idx])
            opt.observe(rec, y_value)
            x_hebo.append(rec.values[0])
            y_hebo.append(opt.y[-1]) 
        hebo_xs.append(x_hebo)
        hebo_ys.append(y_hebo)
    
    return hebo_xs, hebo_ys

def save_batch(x, y, models, fname, fnmodels):
    with open(fname, 'wb') as f:
        print (f"Saving data {f.name}...")
        np.savez_compressed(f, x=x, y=y)
    
    print (f"Save models {fnmodels}...")
    with open(fnmodels, 'wb') as file:
        dump(models, file)
    #torch.save(models, fnmodels)
    print (f"Save completed.")

def load_batch(filename, model_fn):
    f = np.load(filename)
    with open(model_fn, 'rb') as file:
        models = load(file)
    return f["x"], f["y"], models


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("id")
    args = parser.parse_args()
    
    # Parameters
    batch_size = 32
    sequence_length = 100
    min_num_components = 1
    max_num_components = 8
    min_x_value = 0
    max_x_value = 1 # 100 # 4 * np.pi
    min_dim = 2
    max_dim = 4

    num_hebo_runs = 10
    num_hebo_steps = 100

    hebo_total_xs = []
    hebo_total_ys = []
    for _ in range(batch_size):
        dim = np.random.randint(min_dim, max_dim)
        # generate trigonometric fncs
        fnc_list = gen_trigonometric_fncs(1, dim, min_x_value, max_x_value, sequence_length, min_num_components, max_num_components)

        for _ in range(num_hebo_runs):
            # run hebo
            hebo_xs, hebo_ys = run_hebo(num_hebo_steps, fnc_list, dim, min_x_value, max_x_value)
            hebo_total_xs.append(hebo_xs)
            hebo_total_ys.append(hebo_ys)

    array_l = []
    #max_length = max(len(hebo_total_xs[i][0][0]) for i in range(len(hebo_total_xs)))
    #print (max_length)
    for i in range(len(hebo_total_xs)):
        a = np.stack(hebo_total_xs[i])[0]
        padded_arr = np.pad(a, ((0, 0), (0, max_dim - a.shape[1])), mode='constant')
        array_l.append(padded_arr)
    
    # Convert data to numpy array in the right format
    x_batch = np.stack(array_l)
    y_batch = np.stack(hebo_ys)[0]

    # save batch file
    save_batch(x_batch, y_batch, fnc_list, f"data_{args.id}.npz", f"models_{args.id}.dill")
 
