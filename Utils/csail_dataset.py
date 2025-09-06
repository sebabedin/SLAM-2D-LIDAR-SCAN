'''
@author: seb
'''

import json


DATASET_PATH = "../DataSet/PreprocessedData"
CSAIL_PATH = f"{DATASET_PATH}/csail_gfs"

SHORT_CSAIL_PATH = f"{DATASET_PATH}/short_csail_gfs"

if __name__ == '__main__':
    
    
    with open(CSAIL_PATH, 'r') as f:
        dataset_data = json.load(f)
    
    print(dataset_data.keys())
    print(dataset_data["map"].keys())
    print(len(list(dataset_data["map"].keys())))
    
    stamps = list(dataset_data["map"].keys())
    ordered_stamps = sorted(stamps)
    
    print(ordered_stamps)
    
    short_list = ordered_stamps[:10]
    
    print(short_list)
    
    dataset_croped_data = {}
    map_samples = {}
    for stamp in short_list:
        map_samples[stamp] = dataset_data["map"][stamp]
    dataset_croped_data["map"] = map_samples
    
    with open(SHORT_CSAIL_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset_croped_data, f, ensure_ascii=False, indent=4)
    
    # print(dataset_data)
    
