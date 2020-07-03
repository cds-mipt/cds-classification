import json
import os

json_fname = '/home/cds-y/Datasets/38_Classes_NKBVS + synthetic/Info.json'

#target_path = '/home/cds-y/Datasets/38_Classes_NKBVS + synthetic/train'
target_path = '/home/cds-y/Datasets/38_Classes_NKBVS + synthetic/test'

with open(json_fname, 'r') as f:
    json_dict = json.load(f)

for key, value in json_dict.items():
    print(key, ':', value)
    folder_name_from = os.path.join(target_path,str(value))
    folder_name_to = os.path.join(target_path,str(key))
    os.rename(folder_name_from, folder_name_to)