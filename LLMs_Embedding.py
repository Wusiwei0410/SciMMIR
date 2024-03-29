import torch
from PIL import Image
import transformers
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoProcessor,AutoModelForCausalLM
from src.MyBlip.MyBlip import MyBlip2ForConditionalGeneration, Blip2Model
from tqdm import tqdm
import datasets
import os
import json
import argparse

def process_images(images, image_processor, model_cfg=None):
    if model_cfg is not None:
        image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    else:
        image_aspect_ratio = 'resize'
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    elif image_aspect_ratio == 'resize':
        for image in images:
            max_edge = max(image.size)
            image = image.resize((max_edge, max_edge))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images

def load_json(path):
    f = open(path, 'r', encoding = 'utf-8')
    data = json.load(f)
    f.close()

    return data

def json_save(data, path):
    f = open(path, 'w', encoding = 'utf-8')
    json.dump(data, f)
    f.close()

def get_text_embeddings(model, processor, data, max_length, model_name):
    text_embeddings = []
    texst2index = {}
    with torch.no_grad():
        for line in tqdm(data):
            caption = line['text']
            if model_name in ['mplug_owl2_llama2_7b','llava_v1.5_7b']:
                text_inputs = processor['tokenizer'](caption,return_tensors="pt",padding="longest",max_length=max_length,truncation=True,).to(model.device)
                text_features = model.get_text_features(**text_inputs, output_hidden_states = True)
                text_features = text_features.hidden_states[-1]
                attention_mask = text_inputs['attention_mask']

            elif model_name=='fuyu-8b':
                processor.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                text_inputs = processor(text=caption, return_tensors="pt").to(model.device)
                text_features = model.get_text_features(**text_inputs, output_hidden_states = True)
                text_features = text_features.hidden_states[-1]
                attention_mask = text_inputs['attention_mask']
            
            elif model_name=='llama-adapter':
                device = torch.cuda.current_device()
                with torch.cuda.amp.autocast():
                    token=torch.tensor(model.tokenizer.encode(caption, bos=True, eos=False))
                    padding = max_length - token.shape[0]#padding
                    if padding > 0:
                        token = torch.cat((token, torch.zeros(padding, dtype=torch.int64) - 1))
                    elif padding < 0:
                        token = token[:max_length]
                    attention_mask=token.ge(0)
                    token[~attention_mask] = 0
                    text_inputs=token.unsqueeze(0)
                    text_features=model.get_text_features(text_inputs.to(device))

            elif model_name in ['blip2-flan-t5-xl', 'blip2-flan-t5-xxl']:
                text_inputs = processor( text=caption,padding='max_length', max_length = max_length, return_tensors="pt").to(device="cuda")
                text_features = model.get_text_features(**text_inputs, output_hidden_states = True)
                text_features = text_features.last_hidden_state
                attention_mask = text_inputs['attention_mask']
            
            else:#['blip2-opt-2.7b', 'blip2-opt-6.7b','kosmos']:
                text_inputs = processor( text=caption,padding='max_length', max_length = max_length, return_tensors="pt").to(device="cuda")
                text_features = model.get_text_features(**text_inputs, output_hidden_states = True)
                text_features = text_features.hidden_states[-1]
                attention_mask = text_inputs['attention_mask']
            
            text_features = text_features.squeeze(0)
            attention_mask = attention_mask == 1
            attention_mask  = attention_mask.squeeze(0)
            attention_mask = attention_mask.unsqueeze(1)
            text_features = text_features.cpu() * attention_mask.cpu()
            text_features = text_features.sum(dim = 0) / attention_mask.cpu().sum()
            texst2index[caption] = len(text_embeddings)
            text_embeddings.append(text_features.float())
            text_inputs = text_inputs.to(device="cpu")
            torch.cuda.empty_cache()

            # if caption not in texst2index:
            #     texst2index[caption] = len(texst2index)
        text_embeddings = torch.stack(text_embeddings)

    return text_embeddings, texst2index

def get_image_embeddings(model, processor, data, max_length, fig_path,model_name):
    image_embeddings = []
    image2index = {}
    with torch.no_grad():
        for line in tqdm(data):
            image = line['image']
            caption = line['text']
            if model_name=='llama-adapter':
                with torch.cuda.amp.autocast():
                    device = torch.cuda.current_device()
                    image_inputs = processor(image).unsqueeze(0).to(device,dtype=torch.float16)#, dtype=torch.float16
                    image_features = model.forward_visual(imgs=image_inputs)
            
            elif model_name=='mplug_owl2_llama2_7b':
                with torch.cuda.amp.autocast():
                    max_edge = max(image.size)
                    image = image.resize((max_edge, max_edge)) # Resize here for best performance
                    image_inputs = process_images([image],processor['image_processor']).to(model.device,dtype=torch.float16)
                    image_features = model.encode_images(image_inputs).float()

            elif model_name=='llava_v1.5_7b':
                image_inputs=processor['image_processor'].preprocess(image, return_tensors='pt')['pixel_values'].to(model.device,dtype=torch.float16)
                image_features = model.encode_images(image_inputs).float()

            elif model_name in ['blip2-opt-6.7b', 'blip2-opt-2.7b', 'blip2-flan-t5-xl', 'blip2-flan-t5-xxl','fuyu-8b','kosmos']:
                image_inputs =  processor( images = image, return_tensors="pt").to(model.device)#, dtype=torch.float16
                image_features = model.get_image_features(**image_inputs,output_hidden_states=True)

            image_features = image_features.squeeze(0)
            image_features = image_features.mean(dim=0)
            # image_features = image_features.pooler_output.squeeze(0)
            image2index[caption]=len(image_embeddings)
            image_embeddings.append(image_features.float().cpu())
            image_inputs = image_inputs.to(device="cpu")
            torch.cuda.empty_cache()
            # if caption not in image2index:
            #     image2index[caption] = len(image2index)
        image_embeddings = torch.stack(image_embeddings)
    
    return image_embeddings, image2index


def run(args):
    if os.path.exists(f'{args.save_path}')==False:
        os.mkdir(f'{args.save_path}')
    if os.path.exists(f'{args.save_path}{args.model_name}/') == False:
        os.mkdir(f'{args.save_path}{args.model_name}/')

    if args.model_name == 'blip2-flan-t5-xl':
        model = Blip2Model.from_pretrained("Salesforce/blip2-flan-t5-xl").cuda().float()
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    elif args.model_name == 'blip2-flan-t5-xxl':
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
        model = Blip2Model.from_pretrained("Salesforce/blip2-flan-t5-xxl", torch_dtype=torch.float16, device_map="auto")
    elif args.model_name == 'blip2-opt-2.7b':
        model = Blip2Model.from_pretrained('Salesforce/blip2-opt-2.7b').cuda().float()
        processor = Blip2Processor.from_pretrained('Salesforce/blip2-opt-2.7b')
    elif args.model_name == 'blip2-opt-6.7b':
        model = Blip2Model.from_pretrained('Salesforce/blip2-opt-6.7b', torch_dtype=torch.float16, device_map="auto")
        processor = Blip2Processor.from_pretrained('Salesforce/blip2-opt-6.7b')
    elif args.model_name == 'fuyu-8b':
        from src.MyBlip.MyFuyu import FuyuForCausalLM
        from transformers import FuyuProcessor,CLIPImageProcessor
        model = FuyuForCausalLM.from_pretrained("adept/fuyu-8b",device_map='auto').float()
        processor = FuyuProcessor.from_pretrained("adept/fuyu-8b")
    elif args.model_name == 'llama-adapter':
        import src.LLM_models.llama as llama
        device = "cuda" if torch.cuda.is_available() else "cpu"
        llama_dir = "/your_llama-7B_path"
        model, processor = llama.load("BIAS-7B", llama_dir, llama_type="7B", device=device)
    elif args.model_name == "kosmos":
        from src.LLM_models.My_kosmos2 import Kosmos2ForConditionalGeneration
        from transformers import AutoProcessor,AutoModelForCausalLM
        model = Kosmos2ForConditionalGeneration.from_pretrained("microsoft/kosmos-2-patch14-224").cuda().float()
        processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")

    elif args.model_name == 'mplug_owl2_llama2_7b':
        from src.LLM_models.Mymplug_owl2.builder import load_pretrained_model as load_pretrained_model_for_mymplug
        processor={}
        tokenizer, model, image_processor, context_len = load_pretrained_model_for_mymplug('MAGAer13/mplug-owl2-llama2-7b',None, args.model_name,load_8bit=False, load_4bit=False, device_map="cuda", device="cuda")
        # model=model.float()
        tokenizer.padding_side = 'left'
        tokenizer.pad_token_id = tokenizer.eos_token_id
        processor['tokenizer']=tokenizer
        processor['image_processor']=image_processor

    elif args.model_name == 'llava_v1.5_7b':
        from src.LLM_models.MyLLaVA.builder import load_pretrained_model as load_pretrained_model_for_llava
        processor={}
        tokenizer, model, image_processor, context_len = load_pretrained_model_for_llava('liuhaotian/llava-v1.5-7b', None, args.model_name,device_map="cuda", device="cuda")  
        processor['tokenizer']=tokenizer
        processor['image_processor']=image_processor
    model = model.eval()
    ds_remote = datasets.load_dataset("m-a-p/SciMMIR")
    valid_data = ds_remote['validation']
    test_data = ds_remote['test']
    train_data = ds_remote['train']
    
    
    
    #test_data = test_data.select(range(10))
    #train_data = train_data.select(range(10))
    #valid_data = valid_data.select(range(10))
    if args.candidate_span == 'all_data':
        test_data = datasets.concatenate_datasets([test_data, valid_data, train_data])
        text_embeddings, text2index = get_text_embeddings(model, processor, test_data, args.max_length, args.model_name)
        image_embeddings, image2index = get_image_embeddings(model, processor, test_data, args.max_length, args.fig_path,args.model_name)
    elif args.candidate_span == 'test_split':
        text_embeddings, text2index = get_text_embeddings(model, processor, test_data, args.max_length, args.model_name)
        image_embeddings, image2index = get_image_embeddings(model, processor, test_data, args.max_length, args.fig_path,args.model_name)
    json_save(image2index, f'{args.save_path}{args.model_name}/image2index_test.json')
    torch.save(image_embeddings, f'{args.save_path}{args.model_name}/image_embeddings_test.pt')
    json_save(text2index, f'{args.save_path}{args.model_name}/text2index_test.json')
    torch.save(text_embeddings, f'{args.save_path}{args.model_name}/text_embeddings_test.pt')

#读数据

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="config for SciMMIR Based on LLMs")
    parser.add_argument("--saved_data_path" , type = str , default = "")
    # parser.add_argument("--test_data" , type = str , default = parser.parse_args().saved_data_path + 'test_data.json')
    parser.add_argument("--save_path" , type = str , default = "./data/LLMs_Embeddings/")
    parser.add_argument("--fig_path" , type = str , default = "")
    parser.add_argument("--model_name" , type = str , default = 'fuyu-8b')
    parser.add_argument("--candidate_span" , type = str , default = 'all_data')
    parser.add_argument("--max_length" , type = int , default = 256)
    
    # model_name = 'blip2-flan-t5-xl'
    # model_name = 'blip2-opt-2.7b'
    # model_name = 'blip2-opt-6.7b'
    # model_name = 'blip2-flan-t5-xxl'
    # 'fuyu-8b'
    # 'llama-adapter'
    # 'kosmos'
    # 'mplug_owl2_llama2_7b'
    # 'llava_v1.5_7b'
    
    args = parser.parse_args()

    run(args)
