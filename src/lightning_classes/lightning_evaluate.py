'''
Author: kang zhu
Date: 2024-01-16 11:08:54
LastEditors: binary_object_01 binary.object.02@gmail.com
LastEditTime: 2024-01-17 08:01:44
FilePath: /MMIR/src/lightning_classes/lightning_evaluate.py
Description: 
Based on target_rank and target_types, calculate various evaluation metrics for the sciMMIR task,
The metrics include MRR (Mean Reciprocal Rank), hit1, hit3, and hit10

'''
import torch
class SciMMIR_eval():
    def __init__(self):
        self.subcategory_list=[]
        # self.target_rank=target_rank
        # self.target_types=target_types
        # self.image_type=image_type
        # self.direction=direction
        # self.select_print=select_print

    def cal_metric(self,target_rank,target_types,image_type):
        torch.cuda.empty_cache()
        count = 0
        MRR = 0
        hit1 = 0
        hit3 = 0
        hit10 = 0
        for items, items_type in zip(target_rank,target_types):
            if image_type == 'overall':
                for k in [1,3,10]:
                    if k == 1:
                        hit1 += (items <= k).sum()
                    if k == 3:
                        hit3 += (items <= k).sum()
                    if k == 10:
                        hit10 += (items <= k).sum()
                    MRR += (1 / items).sum()
                    count += items.size(0)
            else:
                for item, item_type in zip(items, items_type):
                    if item_type==image_type:
                        count += 1
                        MRR += (1 / item)
                        for k in [1,3,10]:
                            if k == 1:
                                hit1 += (items <= k).sum()
                            if k == 3:
                                hit3 += (items <= k).sum()
                            if k == 10:
                                hit10 += (items <= k).sum()
        MRR = MRR / count
        hit1 = hit1 / count
        hit3 = hit3 / count
        hit10 = hit10 / count
        return MRR,hit1,hit3,hit10

        
    def cal_subcatgories_metric(self,target_rank,target_types):
        metric = {}
        for t in ['fig_architecture', 'fig_illustration', 'fig_result', 'table_result', 'table_parameter']:
            metric[t] = {'count':0, 'MRR':0, 'hit1':0, 'hit3': 0, 'hit10': 0}
            metric[t]['MRR'],metric[t]['hit1'],metric[t]['hit3'],metric[t]['hit10']=self.cal_metric(target_rank,target_types,t)
        return metric


    def cal_rank(self,score, index_y):
        index_x = torch.tensor(list(range(score.size(0))))
        golden_score = score[index_x, index_y]
        target_rank = torch.sum(score >= golden_score.unsqueeze(1), dim=1)
        _, index = torch.sort(score, descending = True)

        return target_rank, index
    
    def _print_metric(self,direction,MRR,hit1,hit3,hit10):
        print(f'Validation/MRR_{direction}','{:.2f}'.format(MRR.item()))
        print(f'Validation/hit1_{direction}','{:.2f}'.format(hit1.item()))
        print(f'Validation/hit3_{direction}','{:.2f}'.format(hit3.item()))
        print(f'Validation/hit10_{direction}', '{:.2f}'.format(hit10.item()))
    
    def print_metric(self,target_rank,target_types,direction,select_print):
        if select_print == 'print_all_setting':
            print('all')
            MRR,hit1,hit3,hit10=self.cal_metric(target_rank,target_types,'overall')
            self._print_metric(direction,MRR,hit1,hit3,hit10)
            metric=self.cal_subcatgories_metric(target_rank,target_types)
            for t in metric:
                print(f"{t}")
                self._print_metric(direction,metric[t]['MRR'],metric[t]['hit1'],metric[t]['hit3'],metric[t]['hit10'])
        elif select_print in ['fig_architecture', 'fig_illustration', 'fig_result', 'table_result', 'table_parameter']:
            print('all')
            MRR,hit1,hit3,hit10=self.cal_metric(target_rank,target_types,'overall')
            metric=self.cal_subcatgories_metric(target_rank,target_types)
            for t in metric:
                if t == select_print:
                    print(f"{t}")
                    self._print_metric(direction,metric[t]['MRR'],metric[t]['hit1'],metric[t]['hit3'],metric[t]['hit10'])

        elif select_print == 'only_overall':
            print('all')
            MRR,hit1,hit3,hit10=self.cal_metric(target_rank,target_types,'overall')
            self._print_metric(direction,MRR,hit1,hit3,hit10)

    # def cal_metric(target_rank, target_types, image_type, direction, select_print = None):
    #     torch.cuda.empty_cache()
    #     count = 0
    #     MRR = 0
    #     hit1 = 0
    #     hit3 = 0
    #     hit10 = 0
    #     for items, items_type in zip(target_rank, target_types):
    #         for k in [1,3,10]:
    #             if k == 1:
    #                 hit1 += (items <= k).sum()
    #             if k == 3:
    #                 hit3 += (items <= k).sum()
    #             if k == 10:
    #                 hit10 += (items <= k).sum()
    #         MRR += (1 / items).sum()
    #         count += items.size(0)
    #     MRR = MRR / count
    #     hit1 = hit1 / count
    #     hit3 = hit3 / count
    #     hit10 = hit10 / count

    #     metric = {}
    #     for t in ['fig_architecture', 'fig_illustration', 'fig_result_fig', 'table_result_tab', 'table_parameter']:
    #         metric[t] = {'count':0, 'MRR':0, 'hit1':0, 'hit3': 0, 'hit10': 0}
    #         for items, items_type in zip(target_rank, target_types):
    #             for item, item_type in zip(items, items_type):
    #                 if item_type == t:
    #                     metric[t]['count'] += 1
    #                     metric[t]['MRR'] += (1 / item)
    #                     for k in [1,3,10]:
    #                         if k == 1:
    #                             metric[t]['hit1'] += (item <= k)
    #                         if k == 3:
    #                             metric[t]['hit3'] += (item <= k)
    #                         if k == 10:
    #                             metric[t]['hit10'] += (item <= k)
    #         metric[t]['MRR'] = metric[t]['MRR'] / metric[t]['count']
    #         metric[t]['hit1'] = metric[t]['hit1'] / metric[t]['count']
    #         metric[t]['hit3'] = metric[t]['hit3'] / metric[t]['count']
    #         metric[t]['hit10'] = metric[t]['hit10'] / metric[t]['count']

    #     if select_print == 'print_all_setting':
    #         print('all')
    #         print(f'Validation/MRR_forward_{direction}','{:.2f}'.format(MRR.item()))
    #         print(f'Validation/hit1_{direction}','{:.2f}'.format(hit1.item()))
    #         print(f'Validation/hit3_{direction}','{:.2f}'.format(hit3.item()))
    #         print(f'Validation/hit10_{direction}', '{:.2f}'.format(hit10.item()))
    #         for t in metric:
    #             print(f"{t}")
    #             print(f'Validation/MRR_forward_{direction}', '{:.2f}'.format(metric[t]['MRR'].item()))
    #             print(f'Validation/hit1_{direction}', '{:.2f}'.format(metric[t]['hit1'].item()))
    #             print(f'Validation/hithit3_{direction}', '{:.2f}'.format(metric[t]['hit3'].item()))
    #             print(f'Validation/hit10_{direction}', '{:.2f}'.format(metric[t]['hit10'].item()))
    #     elif select_print in ['fig_architecture', 'fig_illustration', 'fig_result_fig', 'table_result', 'table_parameter']:
    #         print('all')
    #         print(f'Validation/MRR_forward_{direction}','{:.2f}'.format(MRR.item()))
    #         print(f'Validation/hit1_{direction}','{:.2f}'.format(hit1.item()))
    #         print(f'Validation/hit3_{direction}','{:.2f}'.format(hit3.item()))
    #         print(f'Validation/hit10_{direction}', '{:.2f}'.format(hit10.item()))
    #         for t in metric:
    #             if t == select_print:
    #                 print(f"{t}")
    #                 print(f'Validation/MRR_forward_{direction}', '{:.2f}'.format(metric[t]['MRR'].item()))
    #                 print(f'Validation/hit1_{direction}', '{:.2f}'.format(metric[t]['hit1'].item()))
    #                 print(f'Validation/hit3_{direction}', '{:.2f}'.format(metric[t]['hit3'].item()))
    #                 print(f'Validation/hit10_{direction}', '{:.2f}'.format(metric[t]['hit10'].item()))
    #     elif select_print == 'only_overall':
    #         print('all')
    #         print(f'Validation/MRR_forward_{direction}','{:.2f}'.format(MRR.item()))
    #         print(f'Validation/hit1_{direction}','{:.2f}'.format(hit1.item()))
    #         print(f'Validation/hit3_{direction}','{:.2f}'.format(hit3.item()))
    #         print(f'Validation/hit10_{direction}', '{:.2f}'.format(hit10.item()))
    #     #if select_print don't in these keywords, we don't print results

    #     if image_type == 'overall':
    #         return MRR, hit1, hit3, hit10
    #     else:
    #         return metric[image_type]['MRR'], metric[image_type]['hit1'], metric[image_type]['hit3'], metric[image_type]['hit10']


    # def cal_rank(score, index_y):
    #     index_x = torch.tensor(list(range(score.size(0))))
    #     golden_score = score[index_x, index_y]
    #     target_rank = torch.sum(score >= golden_score.unsqueeze(1), dim=1)
    #     _, index = torch.sort(score, descending = True)

    #     return target_rank, index
