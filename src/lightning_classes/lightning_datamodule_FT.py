#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: Siwei Wu
@file: lightning_datamodule_HRQuery.py 
@time: 2022/08/10
@contact: wusiwei@njust.edu.cn
"""
import logging
import pytorch_lightning as pl
import torch
from src.data_load.data_load_for_FT import SciMMIR_FT_Dataset
from torch.utils.data import DataLoader

class SciMMIR_FT_DataMoudle(pl.LightningDataModule):
    def __init__(self, config , tokenizer, preprocess):
        super(SciMMIR_FT_DataMoudle, self).__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.preprocess = preprocess

    def setup(self, stage = None):
        if stage == 'train' or stage == None:
            self.train_dataset = SciMMIR_FT_Dataset(
                tokenizer = self.tokenizer,
                preprocess = self.preprocess,
                mode = 'train',
                context_length = self.config.context_length,
                text_2_image_index = self.config.datasets_saved_path + self.config.text_2_image_index_valid,
                text_2_index = self.config.datasets_saved_path + self.config.text_2_index_valid,
                config = self.config,
            )

            self.test_dataset = SciMMIR_FT_Dataset(
                tokenizer = self.tokenizer,
                preprocess = self.preprocess,
                mode = 'test',
                context_length = self.config.context_length,
                text_2_image_index = self.config.datasets_saved_path + self.config.text_2_image_index_test,
                text_2_index = self.config.datasets_saved_path + self.config.text_2_index_test,
                config = self.config,
            )

            self.val_dataset = SciMMIR_FT_Dataset(
                tokenizer = self.tokenizer,
                preprocess = self.preprocess,
                mode = 'valid',
                context_length = self.config.context_length,
                text_2_image_index = self.config.datasets_saved_path + self.config.text_2_image_index_valid,
                text_2_index = self.config.datasets_saved_path + self.config.text_2_index_valid,
                config = self.config,
            )

        elif stage == 'test':
            self.test_dataset = SciMMIR_FT_Dataset(
                tokenizer = self.tokenizer,
                preprocess = self.preprocess,
                mode = 'test',
                context_length = self.config.context_length,
                text_2_image_index = self.config.datasets_saved_path + self.config.text_2_image_index_test,
                text_2_index = self.config.datasets_saved_path + self.config.text_2_index_test,
                config = self.config,
            )
    def prepare_data(self):
        logging.info(f"nothing")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size= self.config.train_batch_size,
            num_workers= 32,
            pin_memory= True,
            shuffle= True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.test_batch_size,
            num_workers=16,
            pin_memory=True,
            shuffle=False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.val_batch_size,
            num_workers=16,
            pin_memory=False,
        )

