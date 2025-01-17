import os
import re
import numpy as np
import argparse


def load_batch(filename):
    f = np.load(filename)
    return f["x"], f["y"]


def merge_files(directory, pattern, lb_num, ub_num):
    merged_x = []
    merged_y = []
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".npz"): 
            #print (f"processing {filename}")
            match = re.search(pattern, filename)
            value = int(match.group(1))
            if match and value >= lb_num and value < ub_num:
                x,y = load_batch(
                    os.path.join(directory, filename)
                )
                merged_x.append(x)
                merged_y.append(y)
                number_set.remove(value)
                # print(value)  # Output: 123

    x = np.concatenate(merged_x, axis=0)
    y = np.concatenate(merged_y, axis=0)

    with open(f"data_merged_{lb_num}_{ub_num-1}.npz", 'wb') as f:
        print (f"Saving data {f.name}...")
        np.savez_compressed(f, x=x, y=y)
        print ("Save completed.")

    print ("Missing ids:", ' '.join(list(str(i) for i in number_set)))


if __name__=="__main__":
    #directory = os.fsencode("./output_multi/")
    parser = argparse.ArgumentParser()
    parser.add_argument("lb")
    parser.add_argument("ub")
    args = parser.parse_args()

    directory = "./output_multi/"
    missing_ids = []

    lb_num = int(args.lb)
    ub_num =int(args.ub)

    pattern = r'data_multi_(\d+)\.npz'
    number_set = set(range(lb_num, ub_num))

    merge_files(directory, pattern, lb_num, ub_num)

