#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@author: Siwei Wu
@file: data_load_for_HRQuery.py
@time: 2022/08/10
@contact: wusiwei@njust.edu.cn
"""
import json

import pytorch_lightning as pl
import random
import csv
from torch.utils.data.dataset import Dataset
from dataclasses import dataclass
# from transformers import BertTokenizer , BertModel
import torch
import logging
import copy
from PIL import Image
import datasets
from tqdm import tqdm

@dataclass(frozen = True)
class InputExample:
    Query: str
    Fig: str
    Fig_num: int
    Text_num: int
    Image_type : str

def load_json(path):
    f = open(path , 'r')
    data = json.load(f)
    f.close()

    return data

class SciMMIR_FT_Dataset(Dataset):
    def __init__(
            self,
            tokenizer,
            preprocess,
            mode: str,
            context_length: int,
            text_2_image_index: str,
            text_2_index: str,
            config,
    ):
        super(SciMMIR_FT_Dataset , self).__init__()
        # process = DataProcess()
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        self.mode = mode
        self.context_length = context_length
        self.text_2_image_index = load_json(text_2_image_index)
        self.text_2_index = load_json(text_2_index)
        self.config = config
        self.selected_training_samples_index = {}


        ds_remote = datasets.load_dataset("m-a-p/SciMMIR" )
        if mode == 'train':
            self.examples = ds_remote['train']
            self.examples = self.examples

            if self.config.image_type != 'overall':
                self.selected_training_samples_index = load_json('./data/training_samples_image_type.json')
            logging.info("Current examples: %s" , len(self.examples))

        elif mode == 'test':
            self.examples = ds_remote['test']
            logging.info("Current examples: %s" , len(self.examples))
        elif mode == 'valid':
            self.examples = ds_remote['validation']

            logging.info("Current examples: %s" , len(self.examples))

    def __getitem__(self, index):
        if self.mode == 'train' and self.config.image_type != 'overall':
            example = self.examples[self.selected_training_samples_index[self.config.image_type][str(index)]]
        else:
            example = self.examples[index]
        Query = example['text']
        Fig = example['image']
        if self.mode in ['test', 'valid']:
            Fig_num = self.text_2_image_index[Query]
            Text_num = self.text_2_index[Query]
        else:
            Fig_num = 0
            Text_num = 0
        Image_type = example['class']

        if self.config.Use_BERT == True:
            Q_token = self.tokenizer(Query, add_special_tokens=True,padding='max_length', truncation=True, max_length = self.context_length, return_tensors = 'pt')
        else:
            if self.config.model_name == 'CLIP':
                Q_token = self.tokenizer(Query, context_length=self.context_length, truncate = True)
                Q_token = Q_token.squeeze(1)
            elif self.config.model_name == 'BLIP' or self.config.model_name == 'BLIP-large':
                Q_token = self.preprocess(text = Query, add_special_tokens=True,padding='max_length', truncation=True, max_length = self.context_length, return_tensors = 'pt')

        if self.config.model_name == 'CLIP':
            Fig_preprocess = self.preprocess(Fig).unsqueeze(0)
        elif self.config.model_name == 'BLIP' or self.config.model_name == 'BLIP-large':
            Fig_preprocess = self.preprocess( images = Fig, return_tensors="pt").pixel_values.squeeze(0)#, dtype=torch.float16

        if self.mode == 'train':

            return {
                'Q_token': Q_token,
                'Fig_preprocess': Fig_preprocess,
                'Fig_num': Fig_num,
                'Text_num': Text_num,
                "Image_type": Image_type
            }

        elif self.mode == 'test' or self.mode == 'valid':
            return {
                'Q_token': Q_token,
                'Fig_preprocess': Fig_preprocess,
                'Fig_num': Fig_num,
                'Text_num': Text_num,
                "Image_type": Image_type
            }

    def __len__(self):
        if self.mode == 'train' and self.config.image_type != 'overall':
            # print('subclass: ', len(self.selected_training_samples_index))
            return len(self.selected_training_samples_index[self.config.image_type])
        else:
            return self.examples.num_rows 