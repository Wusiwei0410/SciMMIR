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


class MMIR_LLMs_Dataset(Dataset):
    def __init__(
            self,
            mode: str,
            data_path: str,
            fig_path: str,
            text_2_image_index: str,
            text_2_index: str,
            config,
    ):
        super(MMIR_LLMs_Dataset , self).__init__()
        self.mode = mode
        self.data_path = data_path
        self.fig_path = fig_path
        self.text_2_image_index = load_json(text_2_image_index)
        self.text_2_index = load_json(text_2_index)
        self.config = config
        ds_remote = datasets.load_dataset("m-a-p/SciMMIR")

        if mode == 'test':
            self.examples = ds_remote['test']
            logging.info("Current examples: %s" , len(self.examples))


        logging.info("Current examples: %s" , len(self.examples))

    def __getitem__(self, index):
        example = self.examples[index]
        Query = example['text']
        Fig = example['image']
        Fig_num = self.text_2_image_index[Query]
        Text_num = self.text_2_index[Query]
        Image_type = example['class']

        return {
            'Fig_num': Fig_num,
            'Text_num': Text_num,
            "Image_type": Image_type
        }

    def __len__(self):
        return len(self.examples)
