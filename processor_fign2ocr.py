import jsonlines
import json
from tqdm import tqdm

def load_json(path):
    f = open(path, 'r', encoding = 'utf-8')
    data = json.load(f)
    f.close()

    return data

def json_save(data, path):
    f = open(path, 'w', encoding = 'utf-8')
    json.dump(data, f)
    f.close()

file_jsonl_path = "./data/ocr_result_test.jsonl"

data_raw = []
fign2ocr_dic = {}
with open(file_jsonl_path) as file:
    for line in tqdm(jsonlines.Reader(file)):
        data_raw.append(line)
        fig_filename = line['fig_filename']
        ocr_result = line['ocr_result']
        caption = line['caption']
        fign2ocr_dic[fig_filename] = ocr_result

print(len(data_raw))
print(len(fign2ocr_dic))

json_save(fign2ocr_dic, './data/fign2ocr_dic.json')