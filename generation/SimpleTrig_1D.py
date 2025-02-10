from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import os
from multiprocessing import Process
import multiprocessing
from utils.helper import load_batch, save_batch


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

    fnc_list = []

    # generation loop
    for i in range(batch_size):
        fnc_dict = {}
        
        # Randomize parameters for multiple components
        num_components = np.random.randint(min_num_components, max_num_components)  # Number of sine/cosine components

        amplitudes = []
        frequencies = []
        phses = []
        trigonometrics = []

        for _ in range(num_components):
            amplitude = np.random.uniform(0.05, 1.0)    # Amplitude between 0.5 and 2.0
            frequency = np.random.uniform(5, 25)        # Frequency between 0.5 and 2.0
            phase = np.random.uniform(0, 1)             # Phase shift
            trigonometric = np.sin if np.random.rand() < 0.5 else np.cos

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

    return fnc_list


def run_single_hebo(args):
    """
    Function to run a single HEBO optimization.
    Args:
        args: A tuple containing (fnc, num_hebo_steps).
    """
    fnc, num_hebo_steps = args
    space = DesignSpace().parse([{'name': 'x', 'type': 'num', 'lb': 0, 'ub': 1}])
    opt = HEBO(space)
    x_hebo = []
    y_hebo = []
    for _ in range(num_hebo_steps):
        rec = opt.suggest(n_suggestions=1)
        opt.observe(rec, obj(rec, fnc))
        x_hebo.append(rec.iloc[0].item())
        y_hebo.append(opt.y[-1])
    return x_hebo, y_hebo

def run_hebo_parallel(fnc_list, num_hebo_steps, num_hebo_runs):
    """
    Run HEBO optimization in parallel over both fnc_list and num_hebo_runs.
    """
    hebo_xs = []
    hebo_ys = []
    
    # Prepare the arguments for all HEBO runs
    args = []
    for fnc in fnc_list:
        for _ in range(num_hebo_runs):
            args.append((fnc, num_hebo_steps))
    
    # Create a pool of workers
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Execute all HEBO runs in parallel
        results = list(tqdm(pool.imap(run_single_hebo, args), total=len(args), desc="HEBO Runs"))
    
    # Collect the results
    for x_hebo, y_hebo in results:
        hebo_xs.append(x_hebo)
        hebo_ys.append(y_hebo)
    
    return hebo_xs, hebo_ys

def run_hebo(fnc_list, num_hebo_steps, num_hebo_runs):
    # HEBO loop
    hebo_xs = []
    hebo_ys = []
    for fnc_idx in range(len(fnc_list)):
        print (f"Processing fnc {fnc_idx}...")
        for _ in range(num_hebo_runs):
            space = DesignSpace().parse([{'name' : 'x', 'type' : 'num', 'lb' : 0, 'ub' : 1}])
            opt   = HEBO(space)
            x_hebo = []
            y_hebo = []
            for i in tqdm(range(num_hebo_steps)):
                rec = opt.suggest(n_suggestions = 1)
                opt.observe(rec, obj(rec, fnc_list[fnc_idx]))
                x_hebo.append(rec.iloc[0].item())
                y_hebo.append(opt.y[-1]) 
            hebo_xs.append(x_hebo)
            hebo_ys.append(y_hebo)
    return hebo_xs, hebo_ys

def generate(args):
    # Parameters
    batch_size = args.batchSize
    sequence_length = args.sequenceLength
    min_num_components = args.minComponents
    max_num_components = args.maxComponents
    min_x_value = args.minX
    max_x_value = args.maxX
    num_hebo_runs = args.numHeboTrials
    num_hebo_steps = args.numHeboSteps

    # generate trigonometric fncs
    fnc_list = gen_trigonometric_fncs(batch_size, min_x_value, max_x_value, sequence_length, min_num_components, max_num_components)

    # run hebo
    # hebo_xs, hebo_ys = run_hebo(fnc_list, num_hebo_steps, num_hebo_runs)
    hebo_xs, hebo_ys = run_hebo_parallel(fnc_list, num_hebo_steps, num_hebo_runs)

    # Convert data to numpy array in the right format
    x_batch = np.stack([np.expand_dims(np.array(hebo_xs[i]), 1) for i in range(len(hebo_xs))])
    y_batch = np.stack(hebo_ys)

    # save batch file
    save_batch(x_batch, 
                y_batch, 
                fnc_list, 
                os.path.join("datasets", "single", "1D", f"data_{args.id}.npz"), 
                os.path.join("datasets", "single", "1D", f"models_{args.id}.dill")
    )

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default=0, type=int)
    parser.add_argument("--batchSize", default=1024, type=int)
    parser.add_argument("--sequenceLength", default=100, type=int)
    parser.add_argument("--minComponents", default=2, type=int)
    parser.add_argument("--maxComponents", default=8, type=int)
    parser.add_argument("--minX", default=0, type=int)
    parser.add_argument("--maxX", default=1, type=int)
    parser.add_argument("--numHeboTrials", default=10, type=int)
    parser.add_argument("--numHeboSteps", default=100, type=int)
    args = parser.parse_args()
    
    # Parameters
    batch_size = args.batchSize
    sequence_length = args.sequenceLength
    min_num_components = args.minComponents
    max_num_components = args.maxComponents
    min_x_value = args.minX
    max_x_value = args.maxX
    num_hebo_runs = args.numHeboTrials
    num_hebo_steps = args.numHeboSteps

    # generate trigonometric fncs
    fnc_list = gen_trigonometric_fncs(batch_size, min_x_value, max_x_value, sequence_length, min_num_components, max_num_components)

    # run hebo
    # hebo_xs, hebo_ys = run_hebo(fnc_list, num_hebo_steps, num_hebo_runs)
    hebo_xs, hebo_ys = run_hebo_parallel(fnc_list, num_hebo_steps, num_hebo_runs)

    # Convert data to numpy array in the right format
    x_batch = np.stack([np.expand_dims(np.array(hebo_xs[i]), 1) for i in range(len(hebo_xs))])
    y_batch = np.stack(hebo_ys)

    # save batch file
    save_batch(x_batch, 
                y_batch, 
                fnc_list, 
                os.path.join("datasets", "single", "1D", f"data_{args.id}.npz"), 
                os.path.join("datasets", "single", "1D", f"models_{args.id}.dill")
    )
 