'''
Author: 
Date: 2024-05-22 12:07:23
LastEditors: 
LastEditTime: 2024-10-01 09:59:12
Description: 
'''
import os
import pandas as pd
import numpy as np
import pickle

from datasets import load_dataset
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download

import json
def load_json(file_path):
    with open(file_path, 'r') as f:
        list_data = json.load(f)
    return list_data

def get_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def get_feature_df(data, index):
    df = pd.DataFrame(data)
    if len(index) !=df.shape[0]:
        index = np.concatenate(index).tolist()
        df.index = index
    else:
        df.index = index
    return df

class MoleculeCLADataset(Dataset):
    
    def __init__(self, 
                 target,
                 embed_path, 
                 data_dir, 
                 split_file,
                 task_type, 
                 task_names
                 ):
        self.embed_path = embed_path
        self.file_path = hf_hub_download(repo_id="anonymousxxx/MoleculeCLA", filename=f'labels/{target}.csv', repo_type="dataset")
        self.split_file = hf_hub_download(repo_id="anonymousxxx/MoleculeCLA", filename=f'data.csv', repo_type="dataset")
        
        self.split_dict = {'train':0, 'valid':1, 'test':2}
        self.task_type = task_type
        self.task_names = task_names
        self.dataset_size = -1
        
    def reset_dataset(self,idx):
        self.set_datast_size(len(idx)) 
        self.embed = self.embed.loc[idx].to_numpy()
        self.labels = self.labels.loc[idx].to_numpy()
   
    def set_dataset(self, type):
        
        df = pd.read_csv(self.file_path)
        
        # Get feature
        feature_data = get_pickle_file(self.embed_path)
        self.embed = get_feature_df(feature_data['data'], feature_data['index'])
                        
        # Get split
        splits = pd.read_csv(self.split_file)
        
        # Set dataset
        df = df[splits['scaffold_folds'] == self.split_dict[type]]
        id_list = df['IDs'].values.tolist()
        
        self.embed = self.embed.loc[id_list].to_numpy()
        self.labels = df[self.task_names].values.tolist()
        
        self.set_datast_size(len(self.labels))

    def __len__(self):
        return self.dataset_size
    
    def set_datast_size(self, dataset_size):
        self.dataset_size = dataset_size
    
    def __getitem__(self, idx):
        z = self.embed[idx]
        label = self.labels[idx]
        valid = np.ones_like(label).astype(np.bool_)
        # return z, [], [], label, valid, [], []
        data = {
            'z': z, 
            'pos': None, 
            'pos_target': None,
            'label': label, 
            'valid': valid,
            # 'mol': mol,
            'charges': None
        }
        return data