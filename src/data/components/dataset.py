'''
Author: Jiaxin Zheng
Date: 2024-05-22 12:07:23
LastEditors: Jiaxin Zheng
LastEditTime: 2024-08-16 11:17:30
Description: 
'''
import os
import pandas as pd
import numpy as np
import pickle

from datasets import load_dataset
from torch.utils.data import Dataset

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

class MoleculeCLADatasetLocal(Dataset):
    
    def __init__(self, 
                 target,
                 embed_path, 
                 data_dir, 
                 split_file,
                 task_type, 
                 task_names
                 ):
        self.embed_path = embed_path
        self.file_path = os.path.join(data_dir,f'{target}.csv')
        self.split_file = split_file
        
        self.split_dict = {'train':0, 'valid':1, 'test':2}
        self.task_type = task_type
        self.task_names = task_names
        self.dataset_size = -1
        
    def reset_dataset(self,idx):
        self.set_datast_size(len(idx)) 
        self.embed = self.embed.loc[idx].to_numpy()
        self.labels = self.labels.loc[idx].to_numpy()
   
    def set_dataset(self, type):
        
        df = pd.read_csv(self.file_path, index_col=0)
        
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

def get_label_data(data, splits, target, dataset_type = 'train', task_names = []):
    
    split_dict = {'train':0, 'valid':1, 'test':2}
    dataset_type = split_dict[dataset_type]
    
    target_index_df = pd.DataFrame(data['target'])
    target_index = target_index_df[target_index_df==target].index
    
    scaffold_folds = np.array(splits['scaffold_folds'])
    index = [idx for idx,scaffold_fold in zip(target_index, scaffold_folds) if scaffold_fold == dataset_type]
    data = data[index]
    labels = [data[task_name] for task_name in task_names]
    
    return labels
        
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
        self.target = target
        self.split_file = split_file
        
        self.split_dict = {'train':0, 'valid':1, 'test':2}
        self.task_type = task_type
        self.task_names = task_names
        self.dataset_size = -1
        
    def reset_dataset(self,idx):
        self.set_datast_size(len(idx)) 
        self.embed = self.embed.loc[idx].to_numpy()
        self.labels = self.labels.loc[idx].to_numpy()
   
    def set_dataset(self, type):
        
        data = load_dataset("shikun001/MoleculeCLA", "labels")['train']
        splits = load_dataset("shikun001/MoleculeCLA", "split")['train']
        
        id_list = splits['IDs']
            
        # Get feature
        feature_data = get_pickle_file(self.embed_path)
        self.embed = get_feature_df(feature_data['data'], feature_data['index'])
        
        # Set dataset
        self.labels = get_label_data(data, splits, self.target, type, self.task_names)
        self.embed = self.embed.loc[id_list].to_numpy()
        
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