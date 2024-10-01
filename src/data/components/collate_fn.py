'''
Author: 
Date: 2024-03-28 15:22:40
LastEditors: 
LastEditTime: 2024-06-04 10:14:11
Description: 
'''
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
    
class DownstramTaskPosBatchCollate():
    def __init__(self, is_z_long = True) -> None:
        self.is_z_long = is_z_long
    def __call__(self,data_list):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly."""
        
        keys = ['z', 'pos', 'label', 'valid', 'batch_id', 'mol', 'charges']

        batch = {}
        for key in keys:
            batch[key] = []

        batch_i = 0
        cur_batch_num = len(data_list)
        batch_z = [torch.tensor([])]*cur_batch_num
        batch_pos = [torch.tensor([])]*cur_batch_num
        batch_pos_target = [torch.tensor([])]*cur_batch_num
        batch_label = [torch.tensor([])]*cur_batch_num
        batch_valid = [torch.tensor([])]*cur_batch_num
        batch_charge = [torch.tensor([])]*cur_batch_num
        
        for i,data in enumerate(data_list):
            if data is None:
                continue
            batch_z[i] = torch.tensor(data['z'])
            batch_pos[i] = torch.tensor(data['pos']) if data['pos'] is not None else torch.tensor([])
            batch_pos_target[i] = torch.tensor(data['pos_target']) if data['pos_target'] is not None else torch.tensor([])
            batch_charge[i] = torch.tensor(data['charges']) if 'charges' in data.keys() and data['charges'] is not None else torch.tensor([])
            batch_label[i] = torch.tensor(data['label'])
            batch_valid[i] = torch.tensor(data['valid'])
            batch['batch_id'] = batch['batch_id'] + [batch_i]*len(data['z'])
            batch_i = batch_i+1
            
        if self.is_z_long:
            batch['z'] = torch.cat(batch_z,dim=0).long()
        else:
            batch['z'] = torch.cat(batch_z,dim=0).float()
        batch['pos'] = torch.cat(batch_pos,dim=0).float()
        batch['pos_target'] = torch.cat(batch_pos_target,dim=0).float()
        batch['label'] = torch.cat(batch_label,dim=0)
        batch['valid'] = torch.cat(batch_valid,dim=0).bool()
        batch['batch_id'] = torch.tensor(batch['batch_id'])
        batch['charges'] = torch.cat(batch_charge,dim=0).float()
        
        return batch['z'], batch['pos'], batch['pos_target'], batch['label'], batch['valid'], batch['batch_id'], batch['charges']