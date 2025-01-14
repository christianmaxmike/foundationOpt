import os
import re

directory = os.fsencode("./output_multi/")
    
missing_ids = []

pattern = r'data_multi_(\d+)\.npz'
number_set = set(range(200))



for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".npz") or filename.endswith(".py"): 
        match = re.search(pattern, filename)
        if match:
            value = int(match.group(1))
            # print (value)
            number_set.remove(value)
            # print(value)  # Output: 123

print ("Missing ids:", ' '.join(list(str(i) for i in number_set)))
