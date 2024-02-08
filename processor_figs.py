import json
import torch
import clip
from PIL import Image
from tqdm import tqdm
import os
from transformers import BlipProcessor, Blip2Processor
import argparse
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

def process_fig(data, save_path, preprocess, model_name):
    image_preprocess_data = []
    text_2_image_index = {}
    image_index_2_subclass = {}

    i =  0
    count = 0
    for line in tqdm(data):
        if model_name == 'CLIP':
            image = preprocess(line['image']).unsqueeze(0)
        elif model_name in ['BLIP', 'BLIP-FLAN-T5-XL', 'BLIP-FLAN-T5-XXL']:
            image =  preprocess( images = line['image'], return_tensors="pt").pixel_values
        image_preprocess_data.append(image.cpu())

        text_2_image_index[line['text']] = i
        image_index_2_subclass[i] = line['class']
        if i % 5000 == 4999:
            image_preprocess_data = torch.cat(image_preprocess_data)
            print(image_preprocess_data.size())
            torch.save(image_preprocess_data, f'{save_path}image_preprocess_data_{count}.pt')
            image_preprocess_data = []
            count += 1
        i += 1
    
    if image_preprocess_data :
        image_preprocess_data = torch.cat(image_preprocess_data)
        print(image_preprocess_data.size())
        torch.save(image_preprocess_data, f'{save_path}image_preprocess_data_{count}.pt')
    
    json_save(text_2_image_index, f'{save_path}text_2_image_index.json')
    json_save(image_index_2_subclass, f'{save_path}image_index_2_subclass.json')

def run(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model_name == 'CLIP':
        model, preprocess = clip.load("ViT-B/32", device=device)
    elif args.model_name == 'BLIP':
        preprocess = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", size = args.image_size)
    elif args.model_name == 'BLIP-FLAN-T5-XL':
        preprocess = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", size = args.image_size)
    elif args.model_name == 'BLIP-FLAN-T5-XXL':
        preprocess = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl", size = args.image_size)
    ds_remote = datasets.load_dataset("m-a-p/SciMMIR" )

    if os.path.exists(args.save_path) == False:
        os.mkdir(args.save_path)
    
    #验证集也单独处理一份，这样可以节省验证的事件，验证的时候候选集小一点
    valid_data = ds_remote['validation']
    data = valid_data

    save_path = f'{args.save_path}{args.model_name}_torch_data_valid/'

    if os.path.exists(save_path) == False:
        os.mkdir(save_path)

    process_fig(data, save_path, preprocess, args.model_name)


    #测试集也单独处理一份，这样可以节省验证的事件，验证的时候候选集小一点
    if args.candidates_span == 'all_data':
        test_data = ds_remote['test']
        valid_data = ds_remote['validation']
        train_data = ds_remote['train']
        test_data = datasets.concatenate_datasets([test_data, valid_data, train_data])
    elif args.candidates_span == 'test_split':
        test_data = ds_remote['test']
    data = test_data

    save_path = f'{args.save_path}{args.model_name}_torch_data_test/'

    if os.path.exists(save_path) == False:
        os.mkdir(save_path)

    process_fig(data, save_path, preprocess, args.model_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="config for process text")
    parser.add_argument("--model_name" , type = str , default = "CLIP")
    parser.add_argument("--candidates_span" , type = str , default = "all_data")
    parser.add_argument("--save_path" , type = str , default = './data/')
    parser.add_argument("--image_size" , type = int , default = 384)

    args = parser.parse_args()
    run(args)
