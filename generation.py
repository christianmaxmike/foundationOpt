from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from tqdm import tqdm
import pandas as pd
import numpy as np
from dill import dump, load
import argparse


def obj(params : pd.DataFrame, fnc_dict) -> np.ndarray:
    """
    Load function from the attached function dictionary and return y values of the x values stored in the attached
    parameter (params) dataframe.
    """
    y_values = np.zeros_like(params.values)
    for i in range(fnc_dict['num_components']):
      y_values += fnc_dict['amplitudes'][i] * fnc_dict['trigonometrics'][i](fnc_dict['frequencies'][i] * params.values + fnc_dict['phases'][i])
    return y_values


def gen_trigonometric_fncs(batch_size, min_x_value, max_x_value, sequence_length, min_num_components, max_num_components):
    # Generate x values for the sequence
    x_values = np.linspace(min_x_value, max_x_value, sequence_length)  # Extend the range for more valleys

    # Initialize the batch array
    batch = np.zeros((batch_size, sequence_length))

    # list containing synthesized trigonometric functions
    fnc_list = []

    # generation loop
    for i in range(batch_size):
        
        fnc_dict = {}
        
        # Randomize parameters for multiple components
        num_components = np.random.randint(min_num_components, max_num_components)  # Number of sine/cosine components
        y_values = np.zeros_like(x_values)

        amplitudes = []
        frequencies = []
        phses = []
        trigonometrics = []

        for _ in range(num_components):
            amplitude = np.random.uniform(0.5, 2.0)  # Amplitude between 0.5 and 2.0
            frequency = np.random.uniform(0.5, 2.0)  # Frequency between 0.5 and 2.0
            phase = np.random.uniform(0, 2 * np.pi)  # Phase shift
            trigonometric = np.sin if np.random.rand() < 0.5 else np.cos
            # Alternate between sine and cosine
            y_values += amplitude * trigonometric(frequency * x_values + phase)

            amplitudes.append(amplitude)
            frequencies.append(frequency)
            phses.append(phase)
            trigonometrics.append(trigonometric)
        
        # add values to function dictionary
        fnc_dict['amplitudes'] = amplitudes
        fnc_dict['frequencies'] = frequencies
        fnc_dict['phases'] = phses
        fnc_dict['trigonometrics'] = trigonometrics
        fnc_dict['num_components'] = num_components
        
        fnc_list.append(fnc_dict)

        # Assign to the batch
        batch[i] = y_values

    # normalize x values to be in the range of 0 and 1
    #x_values = x_values / np.max(x_values)
    #for idx, batch_item in enumerate(batch):
    #    batch[idx] = batch_item / np.max(batch_item) # np.linalg.norm(batch_item)
    return fnc_list


def run_hebo(num_hebo_steps):
    # HEBO loop
    hebo_xs = []
    hebo_ys = []
    for fnc_idx in range(len(fnc_list)):
        print (f"Processing fnc {fnc_idx}...")
        space = DesignSpace().parse([{'name' : 'x', 'type' : 'num', 'lb' : 0, 'ub' : 4*np.pi}])
        opt   = HEBO(space)
        x_hebo = []
        y_hebo = []
        for i in tqdm(range(num_hebo_steps)):
            rec = opt.suggest(n_suggestions = 1)
            opt.observe(rec, obj(rec, fnc_list[fnc_idx]))
            #print (f"x: {rec.iloc[0].item()} y: {opt.y[-1]}")
            #print('After %d iterations, best obj is %.2f' % (i, opt.y.min()))
            x_hebo.append(rec.iloc[0].item())
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
    batch_size = 1024
    sequence_length = 100
    min_num_components = 2
    max_num_components = 10
    min_x_value = 0
    max_x_value = 4 * np.pi

    num_hebo_steps = 10

    # generate trigonometric fncs
    fnc_list = gen_trigonometric_fncs(batch_size, min_x_value, max_x_value, sequence_length, min_num_components, max_num_components)

    # run hebo
    hebo_xs, hebo_ys = run_hebo(num_hebo_steps)

    # Convert data to numpy array in the right format
    x_batch = np.stack([np.expand_dims(np.array(hebo_xs[i]), 1) for i in range(len(hebo_xs))])
    y_batch = np.stack(hebo_ys)

    # save batch file
    save_batch(x_batch, y_batch, fnc_list, f"data_{args.id}.npz", f"models_{args.id}.dill")
 