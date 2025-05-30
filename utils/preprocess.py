import os
import numpy as np
import torch
try:
    from scipy.stats import yeojohnson
except ImportError:
    yeojohnson = None

from dill import load


def load_and_preprocess_data(data_config, sequence_length=None, device=None):
    """
    Loads data from .npz files in a directory, applies a transform (minmax or power)
    separately for X and Y, splits into train/val/test, and returns a dictionary containing:
      - X_train, X_val, X_test (torch.Tensor)
      - y_train, y_val, y_test (torch.Tensor)
      - data_config: a dictionary with keys:
            x_dim, y_dim, x_min, x_max, y_min, y_max, transform_type
    """
    # load dataset
    dataset_path = data_config['dataset_path']
    all_x = []
    all_y = []
    data_information = []
    file_idx = []
    for file in os.listdir(dataset_path):
        if file.endswith("data_0.npz"):
            filepath = os.path.join(dataset_path, file)
            with np.load(filepath) as npz_file:
                part_x = npz_file["x"]
                part_y = npz_file["y"]
                all_x.append(part_x)
                all_y.append(part_y)
                file_idx.append(int(file.split(".")[0].split("_")[1]))
    X = np.concatenate(all_x, axis=0)[:64]  # shape [N, T, D]
    y = np.concatenate(all_y, axis=0)[:64]  # shape [N, T, Ydim] or [N, T]
    print ("X shape ", X.shape)
    print ("y shape ", y.shape)

    ## Delete function with extrema
    if False:
        from model.bar_distribution import BarDistribution
        num_bins = 16
        bar_dist_smoothing = 0
        bar_distribution_x = BarDistribution(
                borders=torch.linspace(0, 1, steps=num_bins + 1),
                smoothing=bar_dist_smoothing,
                ignore_nan_targets=True
            )
        argmin_idx = np.argmin(y, axis=1)
        min_vals = X[np.arange(X.shape[0]),argmin_idx.squeeze()]
        bucket_min_vals = bar_distribution_x.map_to_bucket_idx(torch.tensor(min_vals))
        X = X[~((bucket_min_vals==0) | (bucket_min_vals==num_bins-1)).flatten()]
        y = y[~((bucket_min_vals==0) | (bucket_min_vals==num_bins-1)).flatten()]


    # cleaning trajectories - only better values in the trajectory are retained
    # start_idx = 5
    if False:
        for row in range(y.shape[0]):
            c_best = y[row, start_idx, -1] # np.inf
            for col_idx, tp in enumerate(y[row, start_idx:, -1]):
                if tp.item() < c_best:
                    c_best = tp.item()
                else:
                    X[row, col_idx + start_idx + 1] = np.inf
                    y[row, col_idx + start_idx + 1] = np.inf   
        for row in range(X.shape[0]):
            i = np.argmax(X[row, :, -1])
            for col_idx in range(i+1, X.shape[1], 1):
                if X[row, col_idx, -1] != np.inf: 
                    tmp = X[row, col_idx, -1]
                    X[row, col_idx, -1] = X[row, i, -1]
                    X[row, i, -1] = tmp
                    
                    tmp = y[row, col_idx, -1]
                    y[row, col_idx, -1] = y[row, i, -1]
                    y[row, i, -1] = tmp
                    
                    i = np.argmax(X[row, :, -1])
                    col_idx = i+1

    # get indices of best values
    indices = np.argmin(y, axis=1)
    indices = indices.squeeze()
    X_best = np.broadcast_to((X[np.arange(X.shape[0]), indices, np.newaxis]), X.shape)
    y_best = np.broadcast_to(y[np.arange(y.shape[0]), indices, np.newaxis], y.shape)
    print ("X_best shape", X_best.shape)
    print ("y_best shape", y_best.shape)

    transform_type = data_config.get('transform_type', 'none')
    if transform_type == 'minmax':
        # Process X
        shape_x = X.shape
        X_2d = X.reshape(-1, shape_x[-1])
        x_min_arr = X_2d.min(axis=0)
        x_max_arr = X_2d.max(axis=0)
        x_min = float(x_min_arr[0]) if x_min_arr.ndim > 0 else float(x_min_arr)
        x_max = float(x_max_arr[0]) if x_max_arr.ndim > 0 else float(x_max_arr)
        denom = (x_max - x_min) + 1e-10
        X_2d = (X_2d - x_min) / denom
        X = X_2d.reshape(shape_x)

        # Process y
        shape_y = y.shape
        Y_2d = y.reshape(-1, shape_y[-1])
        y_min_arr = Y_2d.min(axis=0)
        y_max_arr = Y_2d.max(axis=0)
        y_min = float(y_min_arr[0]) if y_min_arr.ndim > 0 else float(y_min_arr)
        y_max = float(y_max_arr[0]) if y_max_arr.ndim > 0 else float(y_max_arr)
        denom_y = (y_max - y_min) + 1e-10
        Y_2d = (Y_2d - y_min) / denom_y
        y = Y_2d.reshape(shape_y)

        # For minmax, after normalization, the domain is [0,1]
        x_min, x_max = 0.0, 1.0
        y_min, y_max = 0.0, 1.0

    elif transform_type == 'power':
        if yeojohnson is None:
            raise RuntimeError("scipy.stats not available. Install scipy or remove 'power' transform.")
        shape_x = X.shape
        X_2d = X.reshape(-1, shape_x[-1])
        for d in range(X_2d.shape[1]):
            X_2d[:, d], _ = yeojohnson(X_2d[:, d])
        X = X_2d.reshape(shape_x)
        
        shape_y = y.shape
        Y_2d = y.reshape(-1, shape_y[-1])
        for d in range(Y_2d.shape[1]):
            Y_2d[:, d], _ = yeojohnson(Y_2d[:, d])
        y = Y_2d.reshape(shape_y)
        
        # Derive global bounds from the transformed data.
        x_min = float(X_2d.min())
        x_max = float(X_2d.max())
        y_min = float(Y_2d.min())
        y_max = float(Y_2d.max())
    else:
        shape_x = X.shape
        X_2d = X.reshape(-1, shape_x[-1])
        x_min = float(X_2d.min())
        x_max = float(X_2d.max())
        shape_y = y.shape
        Y_2d = y.reshape(-1, shape_y[-1])
        y_min = float(Y_2d.min())
        y_max = float(Y_2d.max())

    models_path = data_config['dataset_path']
    all_models = [None]*len(file_idx)
    for file in os.listdir(models_path):
        if file.endswith("models_0.dill"):
            filepath = os.path.join(models_path, file)
            with open(filepath, "rb") as f:
                fidx = int(file.split(".")[0].split("_")[1])
                all_models[file_idx.index(fidx)] = load(f)

        
    # Split into train/val/test.
    train_ratio = data_config.get('train_ratio', 0.8)
    val_ratio = data_config.get('val_ratio', 0.1)
    test_ratio = data_config.get('test_ratio', 0.1)
    N = X.shape[0]
    train_size = int(train_ratio * N)
    val_size = int(val_ratio * N)


    #
    flattened_all_models = [xe for x in all_models for xe in x]
    #no_models_train = train_size//len(all_models[0])
    #train_models = all_models[:no_models_train]
    #val_models = all_models[no_models_train:(no_models_train + val_size//len(all_models[0]))]
    #train_models = [xe for x in train_models for xe in x]
    #val_models = [xe for x in val_models for xe in x]
    train_models = flattened_all_models[:train_size]
    val_models =  flattened_all_models[train_size:train_size+val_size]

    # shuffle dataset
    indices = np.arange(N)
    #np.random.shuffle(indices)
    
    # reshuffle x, y datasets
    X = X[indices]
    y = y[indices]
    #y_best = np.broadcast_to(np.min(y, axis=1, keepdims=True), y.shape)
    #y_best_argmin = np.broadcast_to(np.min(y, axis=1, keepdims=True), y.shape)
    X_best = X_best[indices]
    y_best = y_best[indices]

    # extract training set
    X_train = X[:train_size]
    X_best_train = X_best[:train_size]
    y_train = y[:train_size]
    y_train_best = y_best[:train_size]
    
    # extract validation set
    X_val = X[train_size:train_size+val_size]
    X_best_val = X_best[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    y_val_best = y_best[train_size:train_size+val_size]
    
    # extract test datset
    X_test = X[train_size + val_size:]
    X_best_test = X_best[train_size + val_size:]
    y_test = y[train_size + val_size:]
    y_test_best = y[train_size + val_size:]

    if sequence_length is not None:
        X_train = X_train[:, :sequence_length]
        X_val = X_val[:, :sequence_length]
        X_test = X_test[:, :sequence_length]
        X_best_train = X_best_train[:, :sequence_length]
        X_best_val = X_best_val[:, :sequence_length]
        X_best_test = X_best_test[:, :sequence_length]
        if X_train.ndim == y_train.ndim:
            y_train = y_train[:, :sequence_length]
            y_val = y_val[:, :sequence_length]
            y_test = y_test[:, :sequence_length]
    
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    X_best_train = torch.tensor(X_best_train, dtype=torch.float32, device=device)
    X_best_val = torch.tensor(X_best_val, dtype=torch.float32, device=device)
    X_best_test = torch.tensor(X_best_test, dtype=torch.float32, device=device)

    y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
    y_val = torch.tensor(y_val, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.float32, device=device)

    y_train_best = torch.tensor(y_train_best, dtype=torch.float32, device=device)
    y_val_best = torch.tensor(y_val_best, dtype=torch.float32, device=device)
    y_test_best = torch.tensor(y_test_best, dtype=torch.float32, device=device)
 
    if False:
        trunc_param = 16
        chunk_size = 8
        X_train = X_train[:, :trunc_param, :].unfold(dimension = 1,size = chunk_size, step = 1).permute(0,1,3,2).reshape(-1, chunk_size, 1)
        X_val = X_val[:, :trunc_param, :].unfold(dimension = 1,size = chunk_size, step = 1).permute(0,1,3,2).reshape(-1, chunk_size, 1)
        X_test = X_test[:, :trunc_param, :].unfold(dimension = 1,size = chunk_size, step = 1).permute(0,1,3,2).reshape(-1, chunk_size, 1)
        X_best_train = X_best_train[:, :trunc_param, :].unfold(dimension = 1,size = chunk_size, step = 1).permute(0,1,3,2).reshape(-1, chunk_size, 1)
        X_best_val = X_best_val[:, :trunc_param, :].unfold(dimension = 1,size = chunk_size, step = 1).permute(0,1,3,2).reshape(-1, chunk_size, 1)
        X_best_test = X_best_test[:, :trunc_param, :].unfold(dimension = 1,size = chunk_size, step = 1).permute(0,1,3,2).reshape(-1, chunk_size, 1)
        
        y_train = y_train[:, :trunc_param, :].unfold(dimension=1, size=chunk_size, step=1).permute(0,1,3,2).reshape(-1, chunk_size, 1)
        y_val = y_val[:, :trunc_param, :].unfold(dimension=1, size=chunk_size, step=1).permute(0,1,3,2).reshape(-1, chunk_size, 1)
        y_test = y_test[:, :trunc_param, :].unfold(dimension=1, size=chunk_size, step=1).permute(0,1,3,2).reshape(-1, chunk_size, 1)
        y_train_best = y_train_best[:, :trunc_param, :].unfold(dimension=1, size=chunk_size, step=1).permute(0,1,3,2).reshape(-1, chunk_size, 1)
        y_val_best = y_val_best[:, :trunc_param, :].unfold(dimension=1, size=chunk_size, step=1).permute(0,1,3,2).reshape(-1, chunk_size, 1)
        y_test_best = y_test_best[:, :trunc_param, :].unfold(dimension=1, size=chunk_size, step=1).permute(0,1,3,2).reshape(-1, chunk_size, 1)

    print(f"After preprocessing:")
    print(f"  X_train: {X_train.min().item():.4f} / {X_train.max().item():.4f}")
    print(f"  y_train: {y_train.min().item():.4f} / {y_train.max().item():.4f}")
    
    # Assume X shape: [N, T, x_dim] and y shape: [N, T, y_dim] (or [N, T] implies y_dim=1)
    x_dim = X_train.shape[-1]
    y_dim = y_train.shape[-1] if y_train.ndim > 2 else 1
    data_config_out = {
        "x_dim": x_dim,
        "y_dim": y_dim,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "transform_type": transform_type
    }
    
    return {
        "X_train": X_train,
        "X_val": X_val,
        "X_test": X_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "y_train_best": y_train_best,
        "y_test_best": y_test_best,
        "y_val_best": y_val_best,
        "X_train_best": X_best_train,
        "X_test_best": X_best_test,
        "X_val_best": X_best_val,
        "data_config": data_config_out,
        "indices": indices,
        "train_size": train_size,
        "val_size": val_size,
        "all_models": all_models,
        "train_models": train_models,
        "val_models": val_models
    }
