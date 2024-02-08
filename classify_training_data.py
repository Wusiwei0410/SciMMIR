import os
import json
from tqdm import tqdm
import random
import shutil
import datasets
def load_json(path):
    f = open(path, 'r', encoding = 'utf-8')
    data = json.load(f)
    f.close()

    return data

def json_save(data, path):
    f = open(path, 'w', encoding = 'utf-8')
    json.dump(data, f)
    f.close()

 
def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

if __name__ == '__main__':
    ds_remote = datasets.load_dataset("m-a-p/SciMMIR" )
    image_type = 'fig_architecture'
    count = 0
    selected_training_samples_index = {
        'fig_architecture': {},
        'fig_illustration': {},
        'fig_result': {},
        'table_result': {},
        'table_parameter': {},
    }
    
    count = 0
    
    if os.path.exists('./data/') == False:
        os.mkdir('./data/')
    
    #if os.path.exists('./data/training_samples_image_type.json') == False:
    #    os.mkdir('./data/training_samples_image_type.json')

    for line in tqdm(ds_remote['train']):
        selected_training_samples_index[line['class']][len(selected_training_samples_index[line['class']])] = count
        count += 1
    json_save(selected_training_samples_index, './data/training_samples_image_type.json')
