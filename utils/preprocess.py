import os
import numpy as np
import torch
try:
    from scipy.stats import yeojohnson
except ImportError:
    yeojohnson = None


def load_and_preprocess_data(data_config, sequence_length=None, device=None):
    """
    Loads data from .npz files in a directory, applies a transform (minmax or power)
    separately for X and Y, splits into train/val/test, and returns a dictionary containing:
      - X_train, X_val, X_test (torch.Tensor)
      - y_train, y_val, y_test (torch.Tensor)
      - data_config: a dictionary with keys:
            x_dim, y_dim, x_min, x_max, y_min, y_max, transform_type
    """
    dataset_path = data_config['dataset_path']
    all_x = []
    all_y = []
    for file in os.listdir(dataset_path):
        if file.endswith(".npz"):
            filepath = os.path.join(dataset_path, file)
            with np.load(filepath) as npz_file:
                part_x = npz_file["x"]
                part_y = npz_file["y"]
                all_x.append(part_x)
                all_y.append(part_y)
    X = np.concatenate(all_x, axis=0)  # shape [N, T, D]
    y = np.concatenate(all_y, axis=0)  # shape [N, T, Ydim] or [N, T]
    
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
    
    # Split into train/val/test.
    train_ratio = data_config.get('train_ratio', 0.8)
    val_ratio   = data_config.get('val_ratio', 0.1)
    test_ratio  = data_config.get('test_ratio', 0.1)
    N = X.shape[0]
    train_size = int(train_ratio * N)
    val_size = int(val_ratio * N)
    indices = np.arange(N)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]
    
    if sequence_length is not None:
        X_train = X_train[:, :sequence_length]
        X_val = X_val[:, :sequence_length]
        X_test = X_test[:, :sequence_length]
        if X_train.ndim == y_train.ndim:
            y_train = y_train[:, :sequence_length]
            y_val = y_val[:, :sequence_length]
            y_test = y_test[:, :sequence_length]
    
    X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_val = torch.tensor(X_val, dtype=torch.float32, device=device)
    X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_train = torch.tensor(y_train, dtype=torch.float32, device=device)
    y_val = torch.tensor(y_val, dtype=torch.float32, device=device)
    y_test = torch.tensor(y_test, dtype=torch.float32, device=device)
    
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
        "data_config": data_config_out
    }
