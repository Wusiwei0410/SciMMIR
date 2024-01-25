import torch
import json

from tqdm import tqdm

def load_json(path):
    f = open(path, 'r')
    data = json.load(f)
    f.close()

    return data

def cal_error_analysis(index, file_name_index_2_subclass, text_name_index_2_subclass, K, direction = None):
    error_static = {
        'fig_architecture' : {'fig_architecture' : 0, 'fig_illustration' : 0, 'fig_result' : 0, 'table_result' : 0, 'table_parameter' : 0, 'count': 0},
        'fig_illustration' : {'fig_architecture' : 0, 'fig_illustration' : 0, 'fig_result' : 0, 'table_result' : 0, 'table_parameter' : 0, 'count': 0},
        'fig_result' : {'fig_architecture' : 0, 'fig_illustration' : 0, 'fig_result' : 0, 'table_result' : 0, 'table_parameter' : 0, 'count': 0},
        'table_result' : {'fig_architecture' : 0, 'fig_illustration' : 0, 'fig_result' : 0, 'table_result' : 0, 'table_parameter' : 0, 'count': 0},
        'table_parameter' : {'fig_architecture' : 0, 'fig_illustration' : 0, 'fig_result' : 0, 'table_result' : 0, 'table_parameter' : 0, 'count': 0},
        'all' : {'fig_architecture' : 0, 'fig_illustration' : 0, 'fig_result' : 0, 'table_result' : 0, 'table_parameter' : 0, 'count': 0},
    }

    if direction == 'forward':
        for i in range(index.size(0)):
            for k in range(K):
                error_static[text_name_index_2_subclass[str(i)]][file_name_index_2_subclass[str(index[i][k].item())]] += 1
                error_static[text_name_index_2_subclass[str(i)]]['count'] += 1
                error_static['all'][file_name_index_2_subclass[str(index[i][k].item())]] += 1
                error_static['all']['count'] += 1
    elif direction == 'inverse':
        for i in range(index.size(0)):
            for k in range(K):
                error_static[file_name_index_2_subclass[str(i)]][text_name_index_2_subclass[str(index[i][k].item())]] += 1
                error_static[file_name_index_2_subclass[str(i)]]['count'] += 1
                error_static['all'][text_name_index_2_subclass[str(index[i][k].item())]] += 1
                error_static['all']['count'] += 1

    return error_static

def print_error_analysis(error_static, direction):
    print(f'+=========={direction}==========+')
    print('     fig_architecture    fig_illustration    fig_result    table_result    table_parameter')
    for subclss in ['fig_architecture', 'fig_illustration', 'fig_result', 'table_result', 'table_parameter', 'all']:
        print(f"{subclss}: {round(error_static[subclss]['fig_architecture'] / error_static[subclss]['count']* 100, 2) }\
        {round(error_static[subclss]['fig_illustration'] / error_static[subclss]['count'] * 100, 2) }\
        {round(error_static[subclss]['fig_result'] / error_static[subclss]['count']* 100, 2) }\
        {round(error_static[subclss]['table_result'] / error_static[subclss]['count']* 100, 2)}\
        {round(error_static[subclss]['table_parameter'] / error_static[subclss]['count']* 100, 2) }")
        # print(f'{subclss}: {error_static[subclss]['fig_architecture'] / error_static[subclss]['count'] }   {error_static[subclss]['fig_illustration'] / error_static[subclss]['count'] }  {error_static[subclss]['fig_result_fig'] / error_static[subclss]['count'] }   {error_static[subclss]['table_result_tab'] / error_static[subclss]['count'] }   {error_static[subclss]['table_parameter'] / error_static[subclss]['count'] }')

if __name__ == '__main__':
    K = 10
    model_name = 'CLIP'
    forward_index = torch.load("./data/result/ranking_forward.pt")
    inverse_index = torch.load("./data/result/ranking_inverse.pt")

    file_name_index_2_subclass = load_json(f"./data/{model_name}_torch_data_test/text_name_index_2_subclass.json") # because the text is related with the fig
    text_name_index_2_subclass = load_json(f"./data/{model_name}_torch_data_test/text_name_index_2_subclass.json")

    error_static_forward = cal_error_analysis(forward_index, file_name_index_2_subclass, text_name_index_2_subclass, K, direction = 'forward')
    error_static_inverse = cal_error_analysis(inverse_index, file_name_index_2_subclass, text_name_index_2_subclass, K, direction = 'inverse')

    print_error_analysis(error_static_forward, direction = 'forward')
    print_error_analysis(error_static_inverse, direction = 'inverse')

