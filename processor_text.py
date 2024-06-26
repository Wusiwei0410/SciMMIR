import json
import torch
import clip
from PIL import Image
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import os
from transformers import BlipProcessor, BlipModel
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

def process_data(data, tokenizer, model_name, save_path):
    fign2ocr = load_json('./data/fign2ocr_dic.json')
    immage_preprocess_data = []
    text_name_index = {}
    text_name_index_2_subclass = {}
    i =  0
    count = 0
    input_ids = []
    attention_mask = []
    text_token = []

    ocr_input_ids = []
    ocr_attention_mask = []
    ocr_text_token = []
    
    for line in tqdm(data):
        caption = line['text']
        file_name_index = line['file_name_index']
        ocr = fign2ocr[file_name_index]
        # inputs = tokenizer(caption, add_special_tokens=True,padding='max_length', truncation=True, max_length = 128, return_tensors = 'pt')
        if model_name == 'BERT'or model_name in ['BLIP', 'BLIP-FLAN-T5-XXL', 'BLIP-FLAN-T5-XL']:
            text_name_index_2_subclass[len(input_ids)] = line['class']
            text_name_index[caption] = len(input_ids)
            inputs = tokenizer(text = caption, add_special_tokens=True,padding='max_length', truncation=True, max_length = 128, return_tensors = 'pt')
            input_ids.append(inputs['input_ids'])
            attention_mask.append(inputs['attention_mask'])

            inputs = tokenizer(text = ocr, add_special_tokens=True,padding='max_length', truncation=True, max_length = 128, return_tensors = 'pt')
            ocr_input_ids.append(inputs['input_ids'])
            ocr_attention_mask.append(inputs['attention_mask'])
            
        elif model_name  == 'CLIP':
            text_name_index_2_subclass[len(text_token)] = line['class']
            text_name_index[caption] = len(text_token)
            inputs = tokenizer(caption, context_length=77, truncate = True)
            inputs = inputs.squeeze(1)
            text_token.append(inputs)

            inputs = tokenizer(ocr, context_length=77, truncate = True)
            inputs = inputs.squeeze(1)
            ocr_text_token.append(inputs)
    if model_name == 'BERT':
        input_ids = torch.cat(input_ids)
        attention_mask = torch.cat(attention_mask)
        ocr_input_ids = torch.cat(ocr_input_ids)
        ocr_attention_mask = torch.cat(ocr_attention_mask)
        torch.save(input_ids, f'{save_path}input_ids.pt')
        torch.save(attention_mask, f'{save_path}attention_mask.pt')
        torch.save(ocr_input_ids, f'{save_path}ocr_input_ids.pt')
        torch.save(ocr_attention_mask, f'{save_path}ocr_attention_mask.pt')
    elif model_name  == 'CLIP':
        text_token = torch.cat(text_token)
        torch.save(text_token, f'{save_path}text_mat.pt')
        ocr_text_token = torch.cat(ocr_text_token)
        torch.save(ocr_text_token, f'{save_path}ocr_text_mat.pt')
    elif model_name in ['BLIP', 'BLIP-FLAN-T5-XXL', 'BLIP-FLAN-T5-XL']:
        input_ids = torch.cat(input_ids)
        attention_mask = torch.cat(attention_mask)
        torch.save(input_ids, f'{save_path}input_ids.pt')
        torch.save(attention_mask, f'{save_path}attention_mask.pt')

        ocr_input_ids = torch.cat(ocr_input_ids)
        ocr_attention_mask = torch.cat(ocr_attention_mask)
        torch.save(ocr_input_ids, f'{save_path}ocr_input_ids.pt')
        torch.save(ocr_attention_mask, f'{save_path}ocr_attention_mask.pt')
    json_save(text_name_index, f'{save_path}text_index.json')
    json_save(text_name_index_2_subclass, f'{save_path}text_name_index_2_subclass.json')

def run(args):

    ds_remote = datasets.load_dataset("m-a-p/SciMMIR" )
    valid_data = ds_remote['validation']
    test_data = ds_remote['test']

    print(args.model_name)
    if args.model_name ==  'BERT':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased' )
    elif args.model_name  == 'CLIP':
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load("RN50x64", device=device)
        tokenizer = clip.tokenize
    elif args.model_name == 'BLIP':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base" )
        tokenizer = processor
    elif args.model_name == 'BLIP-FLAN-T5-XL':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xl")
        tokenizer = processor
    elif args.model_name == 'BLIP-FLAN-T5-XXL':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        processor = BlipProcessor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
        tokenizer = processor
    
    if os.path.exists(args.save_path) == False:
        os.mkdir(args.save_path)

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
    process_data(data, tokenizer, args.model_name, save_path)


    #验证集也单独处理一份，这样可以节省验证的事件，验证的时候候选集小一点

    data = valid_data
    save_path = f'{args.save_path}{args.model_name}_torch_data_valid/'
    if os.path.exists(save_path) == False:
        os.mkdir(save_path)
    process_data(data, tokenizer, args.model_name, save_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config for process text")
    parser.add_argument("--model_name" , type = str , default = "CLIP")
    parser.add_argument("--save_path" , type = str , default = './data/')
    parser.add_argument("--candidates_span" , type = str , default = "all_data")
    args = parser.parse_args()
    run(args)
