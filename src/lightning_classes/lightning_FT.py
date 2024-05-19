#!/usr/bin/env python  
# -*- coding:utf-8 -*-  
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

def json_save(data, path):
    f = open(path, 'w', encoding = 'utf-8')
    json.dump(data, f)
    f.close()


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
    metric['figure'] = {'count':0, 'MRR':0, 'hit1':0, 'hit3': 0, 'hit10': 0}
    metric['table'] = {'count':0, 'MRR':0, 'hit1':0, 'hit3': 0, 'hit10': 0}
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
                    
                    figure_type = None
                    if 'fig_' in item_type:
                        figure_type = 'figure'
                    elif 'table_' in item_type:
                        figure_type = 'table'
                    
                    metric[figure_type]['count'] += 1
                    metric[figure_type]['MRR'] += (1 / item)
                    for k in [1,3,10]:
                        if k == 1:
                            metric[figure_type]['hit1'] += (item <= k)
                        if k == 3:
                            metric[figure_type]['hit3'] += (item <= k)
                        if k == 10:
                            metric[figure_type]['hit10'] += (item <= k)
        metric[t]['MRR'] = metric[t]['MRR'] / metric[t]['count']
        metric[t]['hit1'] = metric[t]['hit1'] / metric[t]['count']
        metric[t]['hit3'] = metric[t]['hit3'] / metric[t]['count']
        metric[t]['hit10'] = metric[t]['hit10'] / metric[t]['count']

    for figre_type in ['figure', 'table']:
        metric[figre_type]['MRR'] = metric[figre_type]['MRR'] / metric[figre_type]['count']
        metric[figre_type]['hit1'] = metric[figre_type]['hit1'] / metric[figre_type]['count']
        metric[figre_type]['hit3'] = metric[figre_type]['hit3'] / metric[figre_type]['count']
        metric[figre_type]['hit10'] = metric[figre_type]['hit10'] / metric[figre_type]['count']
        
    
    if select_print == 'print_all_setting':
        print('all')
        print(f'Validation/MRR_{direction}','{:.5f}'.format(MRR.item()))
        print(f'Validation/hit1_{direction}','{:.5f}'.format(hit1.item()))
        print(f'Validation/hithit3_{direction}','{:.5f}'.format(hit3.item()))
        print(f'Validation/hit10_{direction}', '{:.5f}'.format(hit10.item()))

        print('Figure')
        print(f'Validation/MRR_{direction}', '{:.5f}'.format(metric['figure']['MRR'].item()))
        print(f'Validation/hit1_{direction}', '{:.5f}'.format(metric['figure']['hit1'].item()))
        print(f'Validation/hithit3_{direction}', '{:.5f}'.format(metric['figure']['hit3'].item()))
        print(f'Validation/hit10_{direction}', '{:.5f}'.format(metric['figure']['hit10'].item()))

        print('Table')
        print(f'Validation/MRR_{direction}', '{:.5f}'.format(metric['table']['MRR'].item()))
        print(f'Validation/hit1_{direction}', '{:.5f}'.format(metric['table']['hit1'].item()))
        print(f'Validation/hithit3_{direction}', '{:.5f}'.format(metric['table']['hit3'].item()))
        print(f'Validation/hit10_{direction}', '{:.5f}'.format(metric['table']['hit10'].item()))

        for t in metric:
            print(f"{t}")
            print(f'Validation/MRR_{direction}', '{:.5f}'.format(metric[t]['MRR'].item()))
            print(f'Validation/hit1_{direction}', '{:.5f}'.format(metric[t]['hit1'].item()))
            print(f'Validation/hithit3_{direction}', '{:.5f}'.format(metric[t]['hit3'].item()))
            print(f'Validation/hit10_{direction}', '{:.5f}'.format(metric[t]['hit10'].item()))
    elif select_print in ['fig_architecture', 'fig_illustration', 'fig_result', 'table_result', 'table_parameter']:
        print('all')
        print(f'Validation/MRR_{direction}','{:.5f}'.format(MRR.item()))
        print(f'Validation/hit1_{direction}','{:.5f}'.format(hit1.item()))
        print(f'Validation/hithit3_{direction}','{:.5f}'.format(hit3.item()))
        print(f'Validation/hit10_{direction}', '{:.5f}'.format(hit10.item()))
        for t in metric:
            if t == select_print:
                print(f"{t}")
                print(f'Validation/MRR_{direction}', '{:.5f}'.format(metric[t]['MRR'].item()))
                print(f'Validation/hit1_{direction}', '{:.5f}'.format(metric[t]['hit1'].item()))
                print(f'Validation/hithit3_{direction}', '{:.5f}'.format(metric[t]['hit3'].item()))
                print(f'Validation/hit10_{direction}', '{:.5f}'.format(metric[t]['hit10'].item()))
    elif select_print == 'only_overall':
        print('all')
        print(f'Validation/MRR_{direction}','{:.5f}'.format(MRR.item()))
        print(f'Validation/hit1_{direction}','{:.5f}'.format(hit1.item()))
        print(f'Validation/hithit3_{direction}','{:.5f}'.format(hit3.item()))
        print(f'Validation/hit10_{direction}', '{:.5f}'.format(hit10.item()))
    #if select_print don't in these keywords, we don't print results

    if image_type == 'overall':
        return MRR, hit1, hit3, hit10
    else:
        return metric[image_type]['MRR'], metric[image_type]['hit1'], metric[image_type]['hit3'], metric[image_type]['hit10']

def cal_rank(score, index_y):
    index_x = torch.tensor(list(range(score.size(0))))
    golden_score = score[index_x, index_y]
    target_rank = torch.sum(score >= golden_score.unsqueeze(1), dim=1)
    _, index = torch.sort(score, descending = True)

    return target_rank, index

def get_embedding(Use_BERT, model_name, datasets_saved_path, text_process_mat_path, figure_process_mat_path, figure_process_mat_num, use_ocr = False, language_model = None, model = None, linear_layer = None, text_process_num = 1000, figure_process_num = 100):
    candidates_image_features = []
    candidates_text_features = []
    candidates_ocr_features = []

    if Use_BERT == True:
        input_ids = torch.load(f'{datasets_saved_path + text_process_mat_path}input_ids.pt')
        attention_mask = torch.load(f'{datasets_saved_path + text_process_mat_path}attention_mask.pt')
        batch_size = input_ids.size(0)
    elif model_name == 'BLIP' or model_name == 'BLIP-large' or model_name == 'BLIP-FLAN-T5-XL' or model_name == 'BLIP-FLAN-T5-XXL':
        input_ids = torch.load(f'{datasets_saved_path + text_process_mat_path}input_ids.pt')
        attention_mask = torch.load(f'{datasets_saved_path + text_process_mat_path}attention_mask.pt')
        batch_size = input_ids.size(0)
    elif model_name == 'CLIP':
        text_token = torch.load(f'{datasets_saved_path + text_process_mat_path}text_mat.pt')
        ocr_token = torch.load(f'{datasets_saved_path + text_process_mat_path}ocr_text_mat.pt')
        batch_size = text_token.size(0)
     
    for i in range(0, batch_size, text_process_num):
        if Use_BERT == True or model_name == 'BLIP' or model_name == 'BLIP-large' or model_name == 'BLIP-FLAN-T5-XL' or model_name == 'BLIP-FLAN-T5-XXL':
            input_ids_item = input_ids[i : i + text_process_num].cuda()
            attention_mask_item = attention_mask[i : i + text_process_num].cuda()
            if Use_BERT == True:
                candidates_text_features.append(linear_layer(language_model(input_ids= input_ids_item, attention_mask = attention_mask_item).last_hidden_state[:,0,:].cuda()))
            elif model_name in ['BLIP-FLAN-T5-XL', 'BLIP-FLAN-T5-XXL']:
                text_feature = model.get_text_features(input_ids= input_ids_item, attention_mask = attention_mask_item, output_hidden_states = True).hidden_states[-1]
                
                attention_mask_item = attention_mask_item.unsqueeze(2)
                text_feature = text_feature * attention_mask_item
                text_feature = text_feature.sum(dim=1) / attention_mask_item.sum(dim=1)
                candidates_text_features.append(text_feature)
            else:
                candidates_text_features.append(model.get_text_features(input_ids= input_ids_item, attention_mask = attention_mask_item))
            input_ids_item = input_ids_item.cpu()
            attention_mask_item = attention_mask_item.cpu()
        else:
            text_token_itme = text_token[i : i + text_process_num].cuda()
            ocr_token_itme = ocr_token[i : i + text_process_num].cuda()
            if model_name == 'CLIP':
                candidates_text_features.append(model.encode_text(text_token_itme.cuda()))
                candidates_ocr_features.append(model.encode_text(ocr_token_itme.cuda()))
            text_token_itme = text_token_itme.cpu()
            ocr_token_itme = ocr_token_itme.cpu()
        torch.cuda.empty_cache()
    candidates_text_features = torch.cat(candidates_text_features, dim = 0)
    if use_ocr == True:
        candidates_ocr_features = torch.cat(candidates_ocr_features, dim = 0)
    else:
        candidates_ocr_features == None

    torch.cuda.empty_cache()

    for i in range(0, figure_process_mat_num):
        figure_process_mat = torch.load(f'{datasets_saved_path + figure_process_mat_path}image_preprocess_data_{i}.pt')
        figure_process_mat =  figure_process_mat
        for j in range(0, figure_process_mat.size(0), figure_process_num):
            figure_process_mat_part = figure_process_mat[j: j + figure_process_num].cuda()
            if model_name == 'CLIP':
                candidates_image_features.append(model.encode_image(figure_process_mat_part.cuda()))
            elif model_name == 'BLIP' or model_name == 'BLIP-large':
                candidates_image_features.append(model.get_image_features(pixel_values = figure_process_mat_part.cuda()))
            elif model_name == 'BLIP-FLAN-T5-XL' or model_name == 'BLIP-FLAN-T5-XXL':
                fig_feature = model.get_image_features(pixel_values = figure_process_mat_part.cuda())
                fig_feature = fig_feature.mean(dim=1)
                candidates_image_features.append(fig_feature)
        torch.cuda.empty_cache()
    figure_process_mat_part = figure_process_mat_part.cpu()
    candidates_image_features = torch.cat(candidates_image_features, dim = 0)
    torch.cuda.empty_cache()

    return candidates_text_features, candidates_image_features, candidates_ocr_features

class SciMMIR_FT(pl.LightningModule):
    def __init__(self, config, model, BERT_model = None):
        super(SciMMIR_FT, self).__init__()
        self.config = config
        self.model = model.float()
        #self.model.train()

        if config.model_name in ['BLIP-FLAN-T5-XXL', 'BLIP-FLAN-T5-XL']:
            for p in self.model.parameters():
                p.requires_grad = False 
            for n, p in self.model.named_parameters():
                if n != 'qformer':
                   p.requires_grad = True 
        
        if BERT_model != None:
            self.BERT = BERT_model
            self.Linear_text = torch.nn.Linear(self.config.text_feature_len, self.config.image_feature_len)
        else:
            self.BERT = None
            self.Linear_text = None
        self.candidates_image_features = None
        self.candidates_text_features = None

        self.target_rank_forward = []
        self.target_rank_inverse = []
        self.ranking_forward = []
        self.ranking_inverse = []
        self.image_type = []
        self.pdist = nn.PairwiseDistance(p=2)

        self.sigmoid = torch.nn.Sigmoid() 
        self.BCE = torch.nn.BCELoss()
        self.all_nodes_features = None
        self.MRR = []
        
    def forward(self, inputs):
        self.model.train()
        Q_token = inputs['Q_token']
        Fig_preprocess = inputs['Fig_preprocess']
        Fig_num = inputs['Fig_num']
        Text_num = inputs['Text_num']
        Image_type = inputs['Image_type']
        Ocr_token = inputs['Ocr_token']

        if self.config.Use_BERT == True:
            Q_token['input_ids'] = Q_token['input_ids'].squeeze(1)
            Q_token['attention_mask'] = Q_token['attention_mask'].squeeze(1)
            outputs = self.BERT(input_ids = Q_token['input_ids'], attention_mask = Q_token['attention_mask'])
            query_features = outputs.last_hidden_state
            query_features = query_features[:,0,:]
            query_features = self.Linear_text(query_features)
        else:
            if self.config.model_name == 'CLIP':
                query_features = self.model.encode_text(Q_token.squeeze(1))
                if self.config.use_ocr == True:
                    ocr_features = self.model.encode_text(Ocr_token.squeeze(1))
            elif self.config.model_name == 'BLIP' or self.config.model_name == 'BLIP-large':
                Q_token['input_ids'] = Q_token['input_ids'].squeeze(1)
                Q_token['attention_mask'] = Q_token['attention_mask'].squeeze(1)
                query_features = self.model.get_text_features(input_ids = Q_token['input_ids'], attention_mask = Q_token['attention_mask'])
            elif self.config.model_name == 'BLIP-FLAN-T5-XL' or self.config.model_name == 'BLIP-FLAN-T5-XXL':
                Q_token['input_ids'] = Q_token['input_ids'].squeeze(1)
                Q_token['attention_mask'] = Q_token['attention_mask'].squeeze(1)
                query_features = self.model.get_text_features(input_ids = Q_token['input_ids'], attention_mask = Q_token['attention_mask'], output_hidden_states = True).hidden_states[-1]
                attention_mask = Q_token['attention_mask']
                attention_mask = attention_mask.unsqueeze(2)
                query_features = query_features * attention_mask
                query_features = query_features.sum(dim=1) / attention_mask.sum(dim=1)
                
        if self.config.model_name == 'CLIP':
            Fig_preprocess = Fig_preprocess.squeeze(1).cuda()
            fig_features = self.model.encode_image(Fig_preprocess)
        elif self.config.model_name == 'BLIP' or self.config.model_name == 'BLIP-large' :
            fig_features = self.model.get_image_features(pixel_values = Fig_preprocess)
        elif self.config.model_name == 'BLIP-FLAN-T5-XL' or self.config.model_name == 'BLIP-FLAN-T5-XXL':
            fig_features = self.model.get_image_features(pixel_values = Fig_preprocess)
            fig_features = fig_features.mean(dim=1)
        
        if self.config.use_ocr == True:
            fig_features = (fig_features + ocr_features) / 2

        if self.candidates_image_features == None or self.candidates_text_features == None:
            score_forward = torch.matmul(query_features, fig_features.transpose(0, 1))
            targets_forward = torch.eye(score_forward.size(0)).cuda()

            score_inverse = torch.matmul(fig_features, query_features.transpose(0, 1))
            targets_inverse = torch.eye(score_inverse.size(0)).cuda()
                
        else:
            score_forward = torch.matmul(query_features, self.candidates_image_features.transpose(0, 1))
            score_inverse = torch.matmul(fig_features, self.candidates_text_features.transpose(0, 1))
            targets_forward = Fig_num
            targets_inverse = Text_num
        return (score_forward, score_inverse, targets_forward, targets_inverse, Image_type)

    def training_step(self, batch, batch_index):
        score_forward, score_inverse, targets_forward, targets_inverse, image_type = self(batch)
        score_forward = self.sigmoid(score_forward)
        score_inverse = self.sigmoid(score_inverse)
        loss_forward = self.BCE(score_forward, targets_forward)
        loss_inverse = self.BCE(score_inverse, targets_inverse)
        # loss = loss_forward
        loss = (loss_forward + loss_inverse) / 2
        self.log("Training/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_index):
        torch.cuda.empty_cache()

        if self.candidates_image_features == None or self.candidates_text_features == None:
            self.model = self.model.eval()
            if self.config.Use_BERT == True:
                self.BERT = self.BERT.eval()
            
            with torch.no_grad():
                candidates_text_features, candidates_image_features, candidates_ocr_features = get_embedding(
                    use_ocr = self.config.use_ocr,
                    Use_BERT = self.config.Use_BERT, 
                    model_name = self.config.model_name, 
                    datasets_saved_path = self.config.datasets_saved_path, 
                    text_process_mat_path = self.config.text_process_mat_path_valid,
                    figure_process_mat_path = self.config.figure_process_mat_path_valid, 
                    figure_process_mat_num = self.config.figure_process_mat_num_valid, 
                    language_model = self.BERT, 
                    model = self.model, 
                    linear_layer = self.Linear_text, 
                    text_process_num = self.config.text_process_num, 
                    figure_process_num = self.config.figure_process_num,
                )
                self.candidates_text_features = candidates_text_features
                self.candidates_image_features = candidates_image_features 

                if self.config.use_ocr == True:
                    self.candidates_image_features = (candidates_image_features + candidates_ocr_features) / 2


        score_forward, score_inverse, index_y_forward, index_y_inverse, image_type = self(batch)
        target_rank_forward, ranking_forward = cal_rank(score_forward, index_y_forward)
        target_rank_inverse, ranking_inverse = cal_rank(score_inverse, index_y_inverse)

        self.target_rank_forward.append(target_rank_forward)
        self.target_rank_inverse.append(target_rank_inverse)
        self.ranking_forward.append(ranking_forward.cpu())
        self.ranking_inverse.append(ranking_inverse.cpu())
        self.image_type.append(image_type)
            

    def on_validation_epoch_end(self):
        MRR, hit1, hit3, hit10 = cal_metric(self.target_rank_forward, self.image_type, self.config.image_type, 'forward', 'don\'t print')
        self.candidates_image_features = None
        self.log('Validation/MRR_forward', MRR, prog_bar=True)

        MRR, hit1, hit3, hit10 = cal_metric(self.target_rank_inverse, self.image_type, self.config.image_type, 'inverse', 'don\'t print')
        self.candidates_text_features = None
        self.log('Validation/MRR_inverse', MRR, prog_bar=True)

        self.target_rank_forward = []
        self.target_rank_inverse = []
        self.ranking_forward  = []
        self.ranking_inverse  = []
        
        self.image_type = []

    def test_step(self, batch, batch_index):

        torch.cuda.empty_cache()
        if self.candidates_image_features == None or self.candidates_text_features == None:
            self.model = self.model.eval()
            if self.config.Use_BERT == True:
                self.BERT = self.BERT.eval()
            
            with torch.no_grad():
                candidates_text_features, candidates_image_features, candidates_ocr_features = get_embedding(
                    use_ocr = self.config.use_ocr,
                    Use_BERT = self.config.Use_BERT, 
                    model_name = self.config.model_name, 
                    datasets_saved_path = self.config.datasets_saved_path, 
                    text_process_mat_path = self.config.text_process_mat_path_test, 
                    figure_process_mat_path = self.config.figure_process_mat_path_test, 
                    figure_process_mat_num = self.config.figure_process_mat_num_test, 
                    language_model = self.BERT, 
                    model = self.model, 
                    linear_layer = self.Linear_text, 
                    text_process_num = self.config.text_process_num, 
                    figure_process_num = self.config.figure_process_num,
                )
                self.candidates_text_features = candidates_text_features
                self.candidates_image_features = candidates_image_features 
                if self.config.use_ocr == True:
                    self.candidates_image_features = (candidates_image_features + candidates_ocr_features) / 2

        score_forward, score_inverse, index_y_forward, index_y_inverse, image_type = self(batch)
        target_rank_forward, ranking_forward = cal_rank(score_forward, index_y_forward)
        target_rank_inverse, ranking_inverse = cal_rank(score_inverse, index_y_inverse)

        self.target_rank_forward.append(target_rank_forward)
        self.target_rank_inverse.append(target_rank_inverse)
        self.ranking_forward.append(ranking_forward.cpu())
        self.ranking_inverse.append(ranking_inverse.cpu())
        self.image_type.append(image_type)

    def on_test_epoch_end(self):
        MRR, hit1, hit3, hit10 = cal_metric(self.target_rank_forward, self.image_type, self.config.image_type, 'forward', self.config.select_print)
        self.candidates_image_features = None
        self.log('Validation/MRR_forward', MRR, prog_bar=True)

        MRR, hit1, hit3, hit10 = cal_metric(self.target_rank_inverse, self.image_type, self.config.image_type, 'inverse', self.config.select_print)
        self.candidates_text_features = None
        self.log('Validation/MRR_inverse', MRR, prog_bar=True)

        self.ranking_forward = torch.cat(self.ranking_forward)
        self.ranking_inverse = torch.cat(self.ranking_inverse)
        

        torch.save(self.ranking_forward,  f"{self.config.result_save_path}ranking_forward.pt")
        torch.save(self.ranking_inverse,  f"{self.config.result_save_path}ranking_inverse.pt")
        
        self.target_rank_forward = []
        self.target_rank_inverse = []
        self.ranking_forward  = []
        self.ranking_inverse  = []
        
        self.image_type = []
        
    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        
        if self.config.Use_BERT == True:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                        "weight_decay": self.config.weight_decay,
                        "lr": self.config.lr_CLIP
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                        "weight_decay": 0,
                        "lr": self.config.lr_CLIP,
                    },
                ]
                optimizer_grouped_parameters.append(
                    {
                        "params": [p for n, p in self.BERT.named_parameters() if not any(nd in n for nd in no_decay)],
                        "weight_decay": self.config.weight_decay,
                        "lr": self.config.lr_BERT
                    }
                )
                optimizer_grouped_parameters.append(
                    {
                        "params": [p for n, p in self.BERT.named_parameters() if any(nd in n for nd in no_decay)],
                        "weight_decay": 0,
                        "lr": self.config.lr_BERT,
                    }
                )

                optimizer_grouped_parameters.append(
                    {
                        "params": self.Linear_text.parameters(),
                        "weight_decay": self.config.weight_decay,
                        "lr": self.config.lr_Linner
                    }
                )
        elif self.config.model_name in ['BLIP-FLAN-T5-XXL', 'BLIP-FLAN-T5-XL']:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in self.model.qformer.named_parameters()],
                        "weight_decay": self.config.weight_decay,
                        "lr": self.config.lr_CLIP
                    },
                ]

        else:
                optimizer_grouped_parameters = [
                    {
                        "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                        "weight_decay": self.config.weight_decay,
                        "lr": self.config.lr_CLIP
                    },
                    {
                        "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                        "weight_decay": 0,
                        "lr": self.config.lr_CLIP,
                    },
                ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            eps=self.config.epsilon,
            weight_decay=self.config.weight_decay
        )

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.num_warmup_steps,
            num_training_steps=self.config.num_training_steps
        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }

        return [optimizer], [scheduler]
