import random
import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm
from hebo.optimizers.hebo import HEBO
from hebo.design_space import DesignSpace
import argparse
import matplotlib.pyplot as plt
import os 
from utils.helper import save_batch, load_batch

class SineGaussianBuilder:

    def __init__(self, n, K, trig_fncs, gausses, ops):
        self.n = n
        self.K = K
        self.trig_fncs = trig_fncs
        self.gausses = gausses
        self.ops = ops

    def sin_or_cos(self, x, trig_fnc_dict):
        trig_fnc = trig_fnc_dict['fnc_type']
        amplitude = trig_fnc_dict['amplitude']
        freq = trig_fnc_dict['freq']
        phase = trig_fnc_dict['phase']
        if trig_fnc == "cos":
            y = amplitude * np.cos(freq * x + phase)
        elif trig_fnc == "sin":
            y = amplitude * np.sin(freq * x + phase)
        else:
            raise NotImplementedError(f"function {trig_fnc} not found.")
        return y

    def gaussian_bell(self, x, gauss_dict):
        mu = gauss_dict['mu']
        sigma = gauss_dict['sigma']
        y = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
        return y

    def combine_two_fncs (self, y1, y2, op):
        if op == "+":
            y_comb = y1 + y2
        elif op == "-":
            y_comb = y1 - y2
        else:
            y_comb = y1 * y2
        return y_comb

    def build_fnc(self, x):
        y_sec = None
        sc_idx = 0
        gauss_idx = 0
        op_idx = 0
        soc = self.sin_or_cos(x, self.trig_fncs[sc_idx])
        gauss = self.gaussian_bell(x, self.gausses[gauss_idx])
        partial_y = self.combine_two_fncs(soc, gauss, self.ops[op_idx])
        print (partial_y[:5])
        sc_idx += 1
        gauss_idx += 1
        op_idx += 1        # print(self.ops)
        for _ in range(self.K):
            soc = self.sin_or_cos(x, self.trig_fncs[sc_idx])
            gauss = self.gaussian_bell(x, self.gausses[gauss_idx])
            pair_y = self.combine_two_fncs(soc, gauss, self.ops[op_idx])
            sc_idx += 1
            gauss_idx += 1
            op_idx += 1
            combined = self.combine_two_fncs(partial_y, pair_y, self.ops[op_idx])
            op_idx += 1
            partial_y = combined
        return partial_y

    def plot_fnc(self, ax):
        domain_start=-3.5 * np.pi
        domain_end=3.5 * np.pi
        domain_points=1000
        x = np.linspace(domain_start, domain_end, domain_points)
        y = self.build_fnc(x) 
        ax.plot(x, y)
        plt.savefig("test.png")
        #plt.show()


class RandomSineGaussianGenerator:
    def __init__(
            self,
            n=10,  # how many final functions to generate
            K=2,  # how many (sin/cos, gaussian) pairs to combine for each function
            domain_start=-3.5 * np.pi,
            domain_end=3.5 * np.pi,
            domain_points=1000,
            seed=42
    ):
        """
        n: number of final functions
        K: how many pairs to combine in each function
        domain_start, domain_end: range for x
        domain_points: number of points for x
        seed: random seed for reproducibility
        """
        self.n = n
        self.K = K
        #np.random.seed(seed)
        #random.seed(seed)
        self.x = np.linspace(domain_start, domain_end, domain_points)

        self.gausses = []
        self.ops = []
        self.trig_fncs = []

    # ------------- Utility Functions ------------- #
    def normalize_array_0_1(self, arr):
        """Map a 1D array to [0,1]. If constant, fill with 0.5."""
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_min == arr_max:  # constant
            return np.full_like(arr, 0.5)
        return (arr - arr_min) / (arr_max - arr_min)

    def limit_absolute_value(self, arr, limit=10):
        """
        Ensure max|arr| <= limit by scaling if necessary.
        This helps keep intermediate values from exploding.
        """
        max_abs = np.max(np.abs(arr))
        if max_abs > limit:
            arr = arr * (limit / max_abs)
        return arr

    # ------------- Base Function Generators ------------- #

    def random_sin_or_cos(self):
        """
        Generate a random sine or cosine with random amplitude, frequency, phase.
        Returns: (y_values, expression_str)
        """
        func_type = random.choice(["sin", "cos"])
        amplitude = random.uniform(0.0, 2.0)
        frequency = random.uniform(0.0, 2.0)
        phase = random.uniform(0, 1)
        if func_type == "sin":
            y = amplitude * np.sin(frequency * self.x + phase)
        else:
            y = amplitude * np.cos(frequency * self.x + phase)
        expr = f"{amplitude:.2f}*{func_type}({frequency:.2f}*x+{phase:.2f})"
        return y, expr, {'fnc_type': func_type,
                "amplitude": amplitude, 
                "freq": frequency,
                "phase": phase}

    def random_gaussian_bell(self):
        """
        Generate a Gaussian bell with max = 1 at x=mu and ~0 at far edges.
        mu in [-3π, 3π], sigma in [0.3, 2], for example.
        Returns: (y_values, expression_str)
        """
        mu = random.uniform(-3 * np.pi, 3 * np.pi)
        sigma = random.uniform(0.5, 1.0)
        # Gaussian: exp(-( (x - mu)^2 ) / (2 * sigma^2))
        y = np.exp(-((self.x - mu) ** 2) / (2 * sigma ** 2))
        expr = f"exp(-((x-{mu:.2f})^2)/(2*{sigma:.2f}^2))"
        return y, expr, {'mu': mu, "sigma": sigma}

    def combine_two_functions(self, f1, f2, op):
        """
        f1, f2: (y, expr) for two functions
        op in {+, -, *}
        Returns: (combined_y, combined_expr)
        """
        y1, expr1 = f1
        y2, expr2 = f2

        if op == '+':
            y_comb = y1 + y2
        elif op == '-':
            y_comb = y1 - y2
        else:  # op == '*'
            y_comb = y1 * y2

        expr_comb = f"({expr1} {op} {expr2})"
        # Limit to avoid large blowups
        # y_comb = self.limit_absolute_value(y_comb, limit=10)
        return y_comb, expr_comb

    # ------------- Combining Logic (K steps) ------------- #

    def random_combined_function(self):
        """
        For K times:
          1) pick a random sin/cos
          2) pick a random Gaussian
          3) combine them with +, -, or *
        Then fold them all together into one final function.

        One way to interpret the request:
        - Each step we produce one partial function from sin-or-cos and gaussian.
        - Then if K>1, we combine this partial with the next partial, etc.

        Alternatively, you could do "chained" combos:
          partial = first pair
          for i in range(K-1):
              next_pair = ...
              partial = partial op next_pair
        Here we'll do the chained approach for clarity.
        """
        sins_coss = []
        gausses = []
        ops = []

        # First pair
        sin_cos = self.random_sin_or_cos()  # (y, expr)
        gauss = self.random_gaussian_bell()  # (y, expr)
        op = random.choice(['+', '-', '*'])
        partial_y, partial_expr = self.combine_two_functions(sin_cos[:-1], gauss[:-1], op)

        sins_coss.append(sin_cos[2])
        gausses.append(gauss[2])
        ops.append(op)

        # If K>1, repeat and combine with the previous partial
        for idx in range(self.K):
            sin_cos2 = self.random_sin_or_cos()
            gauss2 = self.random_gaussian_bell()
            op2 = random.choice(['+', '-', '*'])
            pair_y, pair_expr = self.combine_two_functions(sin_cos2[:-1], gauss2[:-1], op2)

            sins_coss.append(sin_cos2[2])
            gausses.append(gauss2[2])
            ops.append(op2)

            # Now combine partial with pair
            op3 = random.choice(['+', '-', '*'])

            #self.ops.append(op3)
            ops.append(op3)

            combined_y, combined_expr = self.combine_two_functions(
                (partial_y, partial_expr),
                (pair_y, pair_expr),
                op3
            )

            partial_y, partial_expr = combined_y, combined_expr

        return partial_y, partial_expr, sins_coss, gausses, ops


    # ------------- Generating & Plotting ------------- #

    def generate_functions(self):
        """
        Generates n final functions. Normalizes x once to [0,1].
        Normalizes each y to [0,1].
        Returns a list of (x_norm, y_norm, expr) for plotting.
        """
        # Normalize x once
        # x_norm = self.normalize_array_0_1(self.x)

        results = []
        for _ in range(self.n):
            y, expr, sins_coss, gausses, ops = self.random_combined_function()
            self.trig_fncs.append(sins_coss)
            self.gausses.append(gausses)
            self.ops.append(ops)
            # Normalize final y
            # y_norm = self.normalize_array_0_1(y)
            #results.append((x_norm, y_norm, expr))
            results.append((self.x, y, expr))
        return results

    def plot_functions_separate(self, funcs_data, rows=3, cols=4):
        """
        funcs_data: list of (x_norm, y_norm, expr_str)
        Plots each function in its own subplot on [0,1]x[0,1].
        """
        fig, axes = plt.subplots(rows, cols, figsize=(15, 8))
        axes = axes.flatten()

        for ax, (x_norm, y_norm, expr) in zip(axes, funcs_data):
            ax.plot(x_norm, y_norm)
            #ax.set_title(expr, fontsize=8)
            #ax.set_xlim(0, 1)
            #ax.set_ylim(0, 1)
            ax.grid(True)

        # Hide extra subplots if we have fewer funcs than rows*cols
        for ax in axes[len(funcs_data):]:
            ax.set_visible(False)

        plt.tight_layout()
        plt.savefig("test.png")
        return axes

def obj(params : pd.DataFrame, fnc_dict, idx) -> np.ndarray:
    """
    Load function from the attached function dictionary and return y values of the x values stored in the attached
    parameter (params) dataframe.
    """
    fb = SineGaussianBuilder(*fnc_dict)
    return fb.build_fnc(params.values)


def run_single_hebo(args):
    """
    Function to run a single HEBO optimization.
    Args:
        args: A tuple containing (fnc, num_hebo_steps).
    """
    fnc, num_hebo_steps, idx = args
    # space = DesignSpace().parse([{'name': 'x', 'type': 'num', 'lb': 0, 'ub': 1}])
    space = DesignSpace().parse([{'name': 'x', 'type': 'num', 'lb': -3.5 * np.pi, 'ub': 3.5 * np.pi}])
    opt = HEBO(space)
    x_hebo = []
    y_hebo = []
    for _ in range(num_hebo_steps):
        rec = opt.suggest(n_suggestions=1)
        opt.observe(rec, obj(rec, fnc, idx))
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
    for idx, fnc in enumerate(fnc_list):
#        for _ in range(num_hebo_runs):
            print (fnc)
            args.append((fnc, num_hebo_steps, idx))
    
    # Create a pool of workers
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Execute all HEBO runs in parallel
        results = list(tqdm(pool.imap(run_single_hebo, args), total=len(args), desc="HEBO Runs"))
    
    # Collect the results
    for x_hebo, y_hebo in results:
        hebo_xs.append(x_hebo)
        hebo_ys.append(y_hebo)
    
    return hebo_xs, hebo_ys


def test():
    rg = RandomSineGaussianGenerator(n=1, K=30)
    data = rg.generate_functions()
    rg.plot_functions_separate(data, rows=4, cols=5)

    fb = SineGaussianBuilder(rg.n, rg.K, rg.trig_fncs, rg.gausses, rg.ops)
    x_test = np.linspace(0,1,100)
    print(fb.build_fnc(x_test))


def generate(args):
    rg = RandomSineGaussianGenerator(n=args.batchSize, K=30,)
    rg.generate_functions()
    fnc_list = list(zip(np.repeat(rg.n, len(rg.trig_fncs)),
            np.repeat(rg.K, len(rg.trig_fncs)),
            rg.trig_fncs,
            rg.gausses, 
            rg.ops))

    num_hebo_steps = args.numHeboSteps
    num_hebo_runs = args.numHeboTrials
    hebo_xs, hebo_ys = run_hebo_parallel(fnc_list, num_hebo_steps, num_hebo_runs)

    # Convert data to numpy array in the right format
    x_batch = np.stack([np.expand_dims(np.array(hebo_xs[i]), 1) for i in range(len(hebo_xs))])
    y_batch = np.stack(hebo_ys)

    # save batch file
    save_batch(x_batch, 
                y_batch, 
                fnc_list, 
                os.path.join("datasets", "single", "1D_alt", f"data_{args.id}.npz"), 
                os.path.join("datasets", "single", "1D_alt", f"models_{args.id}.dill")
    )
