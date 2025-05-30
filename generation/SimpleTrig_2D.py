from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from ..utils.helper import load_batch, save_batch


def obj(params : pd.DataFrame, fnc_dict) -> np.ndarray:
    """
    Load function from the attached function dictionary and return y values of the x values stored in the attached
    parameter (params) dataframe.
    """
    y_values = np.zeros_like(params.iloc[:, 0].values)
    for i in range(fnc_dict['num_components']):
        y_tmp = np.ones_like(params.iloc[:, 0].values)
        amplitude = fnc_dict['amplitudes'][i]
        for dim_idx , dim in enumerate(range(len(fnc_dict['frequencies'][0]))):
          frequency = fnc_dict['frequencies'][i][dim_idx]
          phase = fnc_dict['phases'][i][dim_idx]
          trigonometric = fnc_dict['trigonometrics'][i][dim_idx]
          y_tmp *= trigonometric(frequency * params.iloc[:, dim_idx].values + phase)
        y_values += amplitude * y_tmp
    return y_values


def gen_trigonometric_fncs(batch_size, dims, min_x_value, max_x_value, sequence_length, min_num_components, max_num_components):
    grid_size = tuple([100]*dims)
    # Generate x values for the sequence
    xs_linspaces = [np.linspace(min_x_value, max_x_value, sequence_length) for _ in range(dims)]
    x_meshgrid = np.meshgrid(*xs_linspaces)
    xs = [*x_meshgrid]
    # Initialize the batch array
    batch = np.zeros((batch_size, *grid_size))

    # list containing synthesized trigonometric functions
    fnc_list = []

    # generation loop
    for i in range(batch_size):
        fnc_dict = {}
        y_values = np.zeros_like(x_meshgrid[0])

        # Number of sine/cosine components
        num_components = np.random.randint(min_num_components, max_num_components)  

        amplitudes = []
        frequencies = []
        phses = []
        trigonometrics = []

        for _ in range(num_components):
            y_tmp = np.ones_like(x_meshgrid[0])
            amplitude = np.random.uniform(0.5, 2.0)
            
            comp_freq = []
            comp_phases = []
            comp_trigonometrics = []

            for dim_idx , dim in enumerate(range(dims)):
                trigonometric = np.sin if np.random.rand() < 0.5 else np.cos
                frequency = np.random.uniform(0.5, 2.0)
                phase = np.random.uniform(0, 2 * np.pi)
                y_tmp *= trigonometric(frequency * xs[dim_idx] + phase)

                comp_freq.append(frequency)
                comp_phases.append(phase)
                comp_trigonometrics.append(trigonometric)

                y_values += amplitude * y_tmp

                frequencies.append(comp_freq)
                phses.append(comp_phases)
                trigonometrics.append(comp_trigonometrics)
                amplitudes.append(amplitude)

        # add values to function dictionary
        fnc_dict['amplitudes'] = amplitudes
        fnc_dict['frequencies'] = frequencies
        fnc_dict['phases'] = phses
        fnc_dict['trigonometrics'] = trigonometrics
        fnc_dict['num_components'] = num_components

        fnc_list.append(fnc_dict)

        # Assign to the batch -> not necessary for later; here: plotting purpose
        batch[i] = y_values
    return fnc_list


def run_hebo(num_hebo_steps, dims):
    hebo_xs = []
    hebo_ys = []
    for fnc_idx in range(len(fnc_list)):
        print (f"Processing fnc {fnc_idx}...")
        space = DesignSpace().parse([{'name' : f'x{idx}', 'type' : 'num', 'lb' : 0, 'ub' : 4*np.pi} for idx in range(dims)])
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
    dims=3
    num_hebo_steps = 10

    
    # generate trigonometric fncs
    fnc_list = gen_trigonometric_fncs(batch_size, dims, min_x_value, max_x_value, sequence_length, min_num_components, max_num_components)

    # run hebo
    hebo_xs, hebo_ys = run_hebo(num_hebo_steps, dims)

    # Convert data to numpy array in the right format
    x_batch = np.stack([np.expand_dims(np.array(hebo_xs[i]), 1) for i in range(len(hebo_xs))])
    y_batch = np.stack(hebo_ys)

    # save batch file
    save_batch(x_batch, y_batch, fnc_list, f"data_{args.id}.npz", f"models_{args.id}.dill")
 