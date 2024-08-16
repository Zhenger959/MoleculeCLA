import numpy as np
from datasets import load_dataset
from src.data.components.dataset import get_label_data

def main():
    targets = ['ADRB2', 'ABL1', 'CYT2C9', 'PPARG', 'GluA2', '3CL', 'HIVINT', 'HDAC2', 'KRAS', 'PDE5']
    task_names = ['docking_score', 'glide_lipo', 'glide_hbond', 'glide_evdw', 'glide_ecoul', 'glide_erotb', 'glide_esite', 'glide_emodel', 'glide_einternal']
    target = targets[0]
    
    data = load_dataset("shikun001/MoleculeCLA", "labels")['train']
    splits = load_dataset("shikun001/MoleculeCLA", "split")['train']
    mol_data = load_dataset("shikun001/MoleculeCLA", "mol_data")['train']
    idx_map = load_dataset("shikun001/MoleculeCLA", "idx_map")['train']
    
    # Get all property labels
    train_labels = get_label_data(data, splits, target, 'train', task_names)
    valid_labels = get_label_data(data, splits, target, 'valid', task_names)
    test_labels = get_label_data(data, splits, target, 'test', task_names)
    
    # Get all molecular information
    id_list = splits['IDs']
    all_index = [idx_map[x] for x in id_list]
    mol_data = np.array(mol_data)[all_index].tolist()

if __name__=='__main__':
    main()