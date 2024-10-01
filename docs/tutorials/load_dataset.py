'''
Author: 
Date: 2024-10-01 09:57:37
LastEditors: 
LastEditTime: 2024-10-01 09:59:01
Description: 
'''
import json

import numpy as np
import pandas as pd

from huggingface_hub import hf_hub_download
from datasets import load_dataset

import pickle

def main():
    targets = ['ADRB2', 'ABL1', 'CYT2C9', 'PPARG', 'GluA2', '3CL', 'HIVINT', 'HDAC2', 'KRAS', 'PDE5']
    task_names = ['docking_score', 'glide_lipo', 'glide_hbond', 'glide_evdw', 'glide_ecoul', 'glide_erotb', 'glide_esite', 'glide_emodel', 'glide_einternal']
    split_dict = {'train':0, 'valid':1, 'test':2}
    
    type = 'train'
    target = targets[5]
    file_path = hf_hub_download(repo_id="anonymousxxx/MoleculeCLA", filename=f'labels/{target}.csv', repo_type="dataset")
    split_file = hf_hub_download(repo_id="anonymousxxx/MoleculeCLA", filename=f'data.csv', repo_type="dataset")
    mol_data_file_path = hf_hub_download(repo_id="anonymousxxx/MoleculeCLA", filename="diversity_molecule_set.pkl", repo_type="dataset")
    idx_map_file_path = hf_hub_download(repo_id="anonymousxxx/MoleculeCLA", filename="docking_id_idx_map.json", repo_type="dataset")
    
    # Get all training property labels
    df = pd.read_csv(file_path)
    splits = pd.read_csv(split_file)
    df = df[splits['scaffold_folds'] == split_dict[type]]
    labels = df[task_names].values.tolist()
    
    # Get all molecular information
    with open(mol_data_file_path, 'rb') as f:
        mol_data = pickle.load(f)
    
    with open(idx_map_file_path, 'r') as f:
        idx_map = json.load(f)
        
    id_list = df['IDs'].values.tolist()
    all_index = [idx_map[x] for x in id_list]
    mol_data = np.array(mol_data)[all_index].tolist()

if __name__=='__main__':
    main()