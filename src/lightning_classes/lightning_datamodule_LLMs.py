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
from src.data_load.data_load_for_LLMs import MMIR_LLMs_Dataset
from torch.utils.data import DataLoader


class MMIR_LLMs_DataMoudle(pl.LightningDataModule):
    def __init__(self, config):
        super(MMIR_LLMs_DataMoudle, self).__init__()
        self.config = config
    
    def setup(self, stage = None):
        if stage == 'test':
            self.test_dataset = MMIR_LLMs_Dataset(
                mode = 'test',
                data_path = self.config.data_path,
                fig_path = self.config.fig_path,
                text_2_image_index = f"{self.config.LLM_text_embedding_saved_path}image2index_test.json",
                text_2_index = f"{self.config.LLM_text_embedding_saved_path}text2index_test.json",
                config = self.config,
            )

    def prepare_data(self):
        logging.info(f"nothing")

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.test_batch_size,
            num_workers=8,
            pin_memory=True,
            shuffle=False
        )

