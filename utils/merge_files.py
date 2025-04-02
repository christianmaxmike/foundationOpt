import os
import re
import numpy as np
import argparse

def load_batch(filename):
    """
    Loads a .npz file and returns:
    - base: x, y
    - power transformed: x_pt, y_pt
    - normalized: x_norm, y_norm
    """
    d = np.load(filename, allow_pickle=True)
    return d["x"], d["y"], d["x_pt"], d["y_pt"], d["x_norm"], d["y_norm"]

def merge_files(directory, pattern, lb_num, ub_num):
    # Lists for merged data for each group
    merged_x = []
    merged_y = []
    merged_x_pt = []
    merged_y_pt = []
    merged_x_norm = []
    merged_y_norm = []

    # Process each file in the directory
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".npz"):
            match = re.search(pattern, filename)
            if match:
                value = int(match.group(1))
                if lb_num <= value < ub_num:
                    file_path = os.path.join(directory, filename)
                    try:
                        x, y, x_pt, y_pt, x_norm, y_norm = load_batch(file_path)
                    except Exception as e:
                        print(f"Error loading {filename}: {e}")
                        continue

                    merged_x.append(x)
                    merged_y.append(y)
                    merged_x_pt.append(x_pt)
                    merged_y_pt.append(y_pt)
                    merged_x_norm.append(x_norm)
                    merged_y_norm.append(y_norm)

                    number_set.remove(value)

    # Concatenate arrays along axis 0
    x_merged = np.concatenate(merged_x, axis=0)
    y_merged = np.concatenate(merged_y, axis=0)
    x_pt_merged = np.concatenate(merged_x_pt, axis=0)
    y_pt_merged = np.concatenate(merged_y_pt, axis=0)
    x_norm_merged = np.concatenate(merged_x_norm, axis=0)
    y_norm_merged = np.concatenate(merged_y_norm, axis=0)

    # Filenames for the three merged files
    base_filename = f"data_merged_{lb_num}_{ub_num-1}.npz"
    pt_filename = f"data_merged_pt_{lb_num}_{ub_num-1}.npz"
    norm_filename = f"data_merged_norm_{lb_num}_{ub_num-1}.npz"
    
    # Save base (raw) data
    with open(base_filename, 'wb') as f:
        print(f"Saving base data to {f.name}...")
        np.savez_compressed(f, x=x_merged, y=y_merged)
        print("Base data save completed.")
    
    # Save power-transformed data
    with open(pt_filename, 'wb') as f:
        print(f"Saving power transformed data to {f.name}...")
        np.savez_compressed(f, x_pt=x_pt_merged, y_pt=y_pt_merged)
        print("Power transformed data save completed.")
    
    # Save normalized data
    with open(norm_filename, 'wb') as f:
        print(f"Saving normalized data to {f.name}...")
        np.savez_compressed(f, x_norm=x_norm_merged, y_norm=y_norm_merged)
        print("Normalized data save completed.")
    
    print("Missing ids:", ' '.join(str(i) for i in sorted(number_set)))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lb", default=0, help="Lower bound for file indices")
    parser.add_argument("--ub", default=100, help="Upper bound (exclusive) for file indices")
    parser.add_argument("--input_folder", help="Folder containing the .npz files")
    args = parser.parse_args()

    directory = args.input_folder
    if not os.path.isdir(directory):
        raise ValueError(f"Directory {directory} does not exist.")
    lb_num = int(args.lb)
    ub_num = int(args.ub)

    pattern = r'data_(\d+)\.npz'
    number_set = set(range(lb_num, ub_num))

    merge_files(directory, pattern, lb_num, ub_num)
