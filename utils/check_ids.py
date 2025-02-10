import os
import re

directory = os.fsencode("./output_multi/")
    
missing_ids = []

pattern = r'data_multi_(\d+)\.npz'
number_set = set(range(1000))

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".npz") or filename.endswith(".py"): 
        match = re.search(pattern, filename)
        if match:
            value = int(match.group(1))
            number_set.remove(value)

print ("Missing ids:", ' '.join(list(str(i) for i in number_set)))
