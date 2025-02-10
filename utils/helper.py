import numpy as np
from dill import dump, load

def save_batch(x, y, models, fname, fnmodels):
    """
    Save batch data and models to files.

    This function saves input data, labels, and models to separate files.
    The data is saved in a compressed NumPy format, while the models are
    serialized using dill.

    Args:
        x (array-like): Input data to be saved.
        y (array-like): Labels or target values to be saved.
        models (object): Models to be saved.
        fname (str): Filename for saving the input data and labels.
        fnmodels (str): Filename for saving the models.

    Returns:
        None

    Raises:
        IOError: If there's an error while writing to the files.
    """
    with open(fname, 'wb') as f:
        print(f"Saving data {f.name}...")
        np.savez_compressed(f, x=x, y=y)
    
    print(f"Save models {fnmodels}...")
    with open(fnmodels, 'wb') as file:
        dump(models, file)
    print("Save completed.")



def load_batch(filename, model_fn):
    """
    Load batch data and models from files.

    This function loads input data, labels, and models from the specified files.
    The data is loaded from a NumPy file, while the models are deserialized
    using dill.

    Args:
        filename (str): Filename of the NumPy file containing input data and labels.
        model_fn (str): Filename of the file containing serialized models.

    Returns:
        tuple: A tuple containing:
            - x (array-like): Loaded input data.
            - y (array-like): Loaded labels or target values.
            - models (object): Loaded models.

    Raises:
        IOError: If there's an error while reading the files.
        ValueError: If the files are corrupted or in an unexpected format.
    """
    f = np.load(filename)
    with open(model_fn, 'rb') as file:
        models = load(file)
    return f["x"], f["y"], models
