#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
""" 
@author: Siwei Wu
@file: lightning_HRQuery.py
@time: 2022/08/10
@contact: wusiwei@njust.edu.cn
"""
import copy
import json
import pytorch_lightning as pl
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.nn as nn
import torch.nn.functional as F

def load_json(path):
    f = open(path, 'r')
    data = json.load(f)
    f.close()

    return data

def cal_metric(target_rank, target_types, image_type, direction, select_print = None):
    torch.cuda.empty_cache()
    count = 0
    MRR = 0
    hit1 = 0
    hit3 = 0
    hit10 = 0
    for items, items_type in zip(target_rank, target_types):
        for k in [1,3,10]:
            if k == 1:
                hit1 += (items <= k).sum()
            if k == 3:
                hit3 += (items <= k).sum()
            if k == 10:
                hit10 += (items <= k).sum()
        MRR += (1 / items).sum()
        count += items.size(0)
    MRR = MRR / count
    hit1 = hit1 / count
    hit3 = hit3 / count
    hit10 = hit10 / count

    metric = {}
    for t in ['fig_architecture', 'fig_illustration', 'fig_result', 'table_result', 'table_parameter']:
        metric[t] = {'count':0, 'MRR':0, 'hit1':0, 'hit3': 0, 'hit10': 0}
        for items, items_type in zip(target_rank, target_types):
            for item, item_type in zip(items, items_type):
                if item_type == t:
                    metric[t]['count'] += 1
                    metric[t]['MRR'] += (1 / item)
                    for k in [1,3,10]:
                        if k == 1:
                            metric[t]['hit1'] += (item <= k)
                        if k == 3:
                            metric[t]['hit3'] += (item <= k)
                        if k == 10:
                            metric[t]['hit10'] += (item <= k)
        metric[t]['MRR'] = metric[t]['MRR'] / metric[t]['count']
        metric[t]['hit1'] = metric[t]['hit1'] / metric[t]['count']
        metric[t]['hit3'] = metric[t]['hit3'] / metric[t]['count']
        metric[t]['hit10'] = metric[t]['hit10'] / metric[t]['count']

    if select_print == 'print_all_setting':
        print('all')
        print(f'Validation/MRR_forward_{direction}','{:.5f}'.format(MRR.item()))
        print(f'Validation/hit1_{direction}','{:.5f}'.format(hit1.item()))
        print(f'Validation/hithit3_{direction}','{:.5f}'.format(hit3.item()))
        print(f'Validation/hit10_{direction}', '{:.5f}'.format(hit10.item()))

        for t in metric:
            print(f"{t}")
            print(f'Validation/MRR_forward_{direction}', '{:.5f}'.format(metric[t]['MRR'].item()))
            print(f'Validation/hit1_{direction}', '{:.5f}'.format(metric[t]['hit1'].item()))
            print(f'Validation/hithit3_{direction}', '{:.5f}'.format(metric[t]['hit3'].item()))
            print(f'Validation/hit10_{direction}', '{:.5f}'.format(metric[t]['hit10'].item()))
    elif select_print in ['fig_architecture', 'fig_illustration', 'fig_result', 'table_result', 'table_parameter']:
        print('all')
        print(f'Validation/MRR_forward_{direction}','{:.5f}'.format(MRR.item()))
        print(f'Validation/hit1_{direction}','{:.5f}'.format(hit1.item()))
        print(f'Validation/hithit3_{direction}','{:.5f}'.format(hit3.item()))
        print(f'Validation/hit10_{direction}', '{:.5f}'.format(hit10.item()))
        for t in metric:
            if t == select_print:
                print(f"{t}")
                print(f'Validation/MRR_forward_{direction}', '{:.5f}'.format(metric[t]['MRR'].item()))
                print(f'Validation/hit1_{direction}', '{:.5f}'.format(metric[t]['hit1'].item()))
                print(f'Validation/hithit3_{direction}', '{:.5f}'.format(metric[t]['hit3'].item()))
                print(f'Validation/hit10_{direction}', '{:.5f}'.format(metric[t]['hit10'].item()))
    elif select_print == 'only_overall':
        print('all')
        print(f'Validation/MRR_forward_{direction}','{:.5f}'.format(MRR.item()))
        print(f'Validation/hit1_{direction}','{:.5f}'.format(hit1.item()))
        print(f'Validation/hithit3_{direction}','{:.5f}'.format(hit3.item()))
        print(f'Validation/hit10_{direction}', '{:.5f}'.format(hit10.item()))

    if image_type == 'overall':
        return MRR, hit1, hit3, hit10
    else:
        return metric[image_type]['MRR'], metric[image_type]['hit1'], metric[image_type]['hit3'], metric[image_type]['hit10']

def cal_rank(score, index_y):
    index_x = torch.tensor(list(range(score.size(0)))).cuda()
    golden_score = score[index_x, index_y]
    target_rank = torch.sum(score >= golden_score.unsqueeze(1), dim=1)
    _, index = torch.sort(score, descending = True)

    return target_rank, index

class MMIR_LLMs(pl.LightningModule):
    def __init__(self, config):
        super(MMIR_LLMs, self).__init__()
        self.eval_model=SciMMIR_eval()
        self.config = config
        self.candidates_image_features = None
        self.candidates_text_features = None

        self.target_rank_forward = []
        self.target_rank_inverse = []
        self.image_type = []
        self.pdist = nn.PairwiseDistance(p=2)

        self.sigmoid = torch.nn.Sigmoid() 
        self.BCE = torch.nn.BCELoss()

        self.all_nodes_features = None
        self.MRR = []
        
    def forward(self, inputs):
        Fig_num = inputs['Fig_num']
        Text_num = inputs['Text_num']
        Image_type = inputs['Image_type']
                
        if self.config.score_method == 'Matrix Dot Product':
            query_features = self.candidates_text_features[Text_num]
            fig_features = self.candidates_image_features[Fig_num]
            score_forward = torch.matmul(query_features, self.candidates_image_features.transpose(0, 1))
            score_inverse = torch.matmul(fig_features, self.candidates_text_features.transpose(0, 1))
        targets_forward = Fig_num
        targets_inverse = Text_num
        return (score_forward, score_inverse, targets_forward, targets_inverse, Image_type)

    def training_step(self, batch, batch_index):
        pass

    def validation_step(self, batch, batch_index):
        pass
            

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch, batch_index):

        torch.cuda.empty_cache()
        if self.candidates_image_features == None or self.candidates_text_features == None:
            self.candidates_text_features = torch.load(f"{self.config.LLM_text_embedding_saved_path}text_embeddings_test.pt").cuda()
            self.candidates_image_features = torch.load(f"{self.config.LLM_text_embedding_saved_path}image_embeddings_test.pt").cuda()

        score_forward, score_inverse, index_y_forward, index_y_inverse, image_type = self(batch)

        target_rank_forward, ranking_forward = cal_rank(score_forward, index_y_forward)
        target_rank_inverse, ranking_inverse = cal_rank(score_inverse, index_y_inverse)

        self.target_rank_forward.append(target_rank_forward)
        self.target_rank_inverse.append(target_rank_inverse)
        self.image_type.append(image_type)

    def on_test_epoch_end(self):
        MRR, hit1, hit3, hit10 = cal_metric(self.target_rank_forward, self.image_type, self.config.image_type, 'forward', self.config.select_print)

        self.candidates_image_features = None
        self.target_rank_forward = []
        self.log('Validation/MRR_forward', MRR, prog_bar=True)

        MRR, hit1, hit3, hit10 = cal_metric(self.target_rank_inverse, self.image_type, self.config.image_type, 'inverse', self.config.select_print)

        self.target_rank_inverse = []
        self.candidates_text_features = None
        self.log('Validation/MRR_inverse', MRR, prog_bar=True)

        self.image_type = []
        
    def configure_optimizers(self):
        pass
