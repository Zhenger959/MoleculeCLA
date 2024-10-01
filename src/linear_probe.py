'''
Author: 
Date: 2024-06-03 20:44:50
LastEditors: 
LastEditTime: 2024-06-04 10:30:44
Description: 
'''
import rootutils
from datetime import datetime
from tqdm import tqdm
import argparse
import os
import pickle
import pandas as pd
import numpy as np
import scipy

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

rootutils.setup_root(__file__,indicator='.project-root',pythonpath=True)

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

def evaluate(y_pred,y_true):
    metric_res = {}
    metric_res['RMSE'] = np.sqrt(np.mean((y_pred-y_true)**2,axis=0))
    metric_res['Pearson'] = scipy.stats.pearsonr(y_pred, y_true)[0]
    metric_res['MAE'] = mean_absolute_error(y_true, y_pred)
    metric_res['R2'] = r2_score(y_true, y_pred)

    return metric_res

def main(args):
    model_name = args.model_name
    file_path = args.input_path
    
    # Model Settings
    model_dict = {
        'Linear_regression': LinearRegression,
        'Lasso': Lasso,
        'Ridge': Ridge,
        'Logistic_regression': LogisticRegression,
        'KNN': KNeighborsRegressor,
        'SVM': SVR,
        'Random_forest': RandomForestRegressor,
        'XGBoost': GradientBoostingRegressor
    }
    
    # Docking Data Setttings
    targets = ["ADRB2","ABL1","CYT2C9","PPARG","GluA2","3CL","HIVINT","HDAC2","KRAS","PDE5"]
    subtask_name = ['docking_score', 'glide_ligand_efficiency', 'glide_ligand_efficiency_sa', 'glide_ligand_efficiency_ln', 'glide_gscore', 'glide_lipo', 'glide_hbond', 'glide_metal', 'glide_rewards', 'glide_evdw', 'glide_ecoul', 'glide_erotb', 'glide_esite', 'glide_emodel', 'glide_energy', 'glide_einternal', 'glide_eff_state_penalty', 'glide_rmsd_to_input']
    split_dict = {'train':0, 'valid':1, 'test':2}
    
    # Feature Data Settings
    label_path = args.split_file  # 140697
    label_dir = args.label_dir
    save_dir = args.output_dir
    
    current_datetime = datetime.now()
    date_string = current_datetime.strftime("%Y%m%d")
    model_folder = model_name+'_'+date_string

    if not os.path.exists(os.path.join(save_dir)):
        os.mkdir(os.path.join(save_dir))
    if not os.path.exists(os.path.join(save_dir,model_folder)):
        os.mkdir(os.path.join(save_dir,model_folder))
    if not os.path.exists(os.path.join(save_dir,model_folder,args.split)):
        os.mkdir(os.path.join(save_dir,model_folder,args.split))
    
    # Results
    expand_name_list = [f'{t}_{s}' for t in targets for s in subtask_name ] + [f'{s}_Avg' for s in subtask_name]
        
    # Get the label data
    label_data = pd.read_csv(label_path)
    
    # Get the train and test index list
    train_index_list = []  # [train_1, train_2] 
    test_index_list = []
    
    if args.split == 'scaffold':
        train_index_list = [np.arange(label_data.shape[0])[label_data['scaffold_folds'] == split_dict['train']]]
        test_index_list = [np.arange(label_data.shape[0])[label_data['scaffold_folds'] == split_dict['test']]]
        
    else:
        kf = KFold(n_splits=5,random_state=42,shuffle=True)
        n_samples = np.arange(label_data.shape[0])
        for train_idx, test_idx in kf.split(n_samples):
            train_index_list.append(train_idx)
            test_index_list.append(test_idx)
            
    # Linear Probe
        
    all_metrics = {}
    metric_list = ['Pearson', 'RMSE', 'MAE', 'R2']
    for metric in metric_list:
        all_metrics[metric] = pd.DataFrame([])
    
    # Load the feature
    feature_name = os.path.basename(file_path).split('.')[0]
    cur_save_dir = os.path.join(save_dir,model_folder, args.split,feature_name)
    if not os.path.exists(cur_save_dir):
        os.mkdir(cur_save_dir)
    print(feature_name)
    
    
    feature_data = get_pickle_file(file_path)
    feature_df = get_feature_df(feature_data['data'], feature_data['index'])
    
    # Train&Test dataset
    for train_idx, test_idx in zip(train_index_list, test_index_list):
        
        all_train_ids_list = label_data.loc[train_idx]['IDs'].values.tolist()
        all_test_idx_list = label_data.loc[test_idx]['IDs'].values.tolist()
        
        train_feature = feature_df.loc[all_train_ids_list].to_numpy()
        test_feature = feature_df.loc[all_test_idx_list].to_numpy()
        
        for target in tqdm(targets):
            dataset = pd.read_csv(os.path.join(label_dir,f'{target}.csv'), index_col=0)
            dataset.index = dataset['IDs']
            
            train_dataset = dataset.loc[all_train_ids_list]
            test_dataset = dataset.loc[all_test_idx_list]
            
            subtask_metric_dict={}
            # print(target)
            
            for subtask in tqdm(subtask_name):
                # print(subtask)
                label_name = f'{subtask}'
                
                train_labels = train_dataset[label_name].values.tolist()
                test_labels = test_dataset[label_name].values.tolist()
                
                if model_name == 'Lasso':
                    model = model_dict[model_name](alpha=0.001)
                else:
                    model = model_dict[model_name]()
                
                model.fit(train_feature, train_labels)
                y_pred = model.predict(test_feature)
                
                subtask_metric_res = evaluate(y_pred, test_labels)
                
                for metric in metric_list:
                    if metric not in subtask_metric_dict:
                        subtask_metric_dict[metric] = {}
                    subtask_metric_dict[metric][subtask] = subtask_metric_res[metric]
                
            for metric in metric_list:
                if target not in all_metrics[metric].keys():
                    # all_metrics[metric][target]=subtask_metric_dict[metric]  # column
                    all_metrics[metric][target] = {k:[v] for k,v in subtask_metric_dict[metric].items()}
                else:
                    metric_dict_list = {}
                    for k, v in subtask_metric_dict[metric].items():
                        metric_dict_list[k] = all_metrics[metric][target][k] + [v]
                    all_metrics[metric][target] = metric_dict_list
    
    for metric in metric_list:
        for k in all_metrics[metric].columns:
            all_metrics[metric][k] = all_metrics[metric][k].apply(lambda x:np.mean(x[0])).values.tolist()
        
    for metric in metric_list:
        all_metrics[metric].to_csv(os.path.join(cur_save_dir, f'{metric}.csv'))
        all_metrics[metric].mean(axis=1).to_csv(os.path.join(cur_save_dir, f'mean_{metric}.csv'))
        
        metric_value = all_metrics[metric].values.T.reshape(-1).tolist() + all_metrics[metric].mean(axis=1).tolist()
        concat_df = pd.DataFrame([metric_value])
        concat_df.columns = expand_name_list
        concat_df.to_csv(os.path.join(cur_save_dir, f'all_{metric}.csv'))
            

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-name', type = str, default='Linear_regression', help = 'Linear probe model')
    parser.add_argument('-s', '--split', type = str, default='scaffold', choices=['scaffold', 'kfold'], help = 'Dataset split method')
    parser.add_argument('-i', '--input_path', type = str, default='data/model_feature/descriptors_3D.pkl', help= 'Representation file path')
    parser.add_argument('-f', '--split_file', type = str, default='data/data.csv', help= 'Split file path')
    parser.add_argument('-l', '--label_dir', type = str, default='data/labels', help= 'Label file directory')
    parser.add_argument('-o', '--output_dir', type = str, default='res', help= 'Label file directory')
    args = parser.parse_args()
    return args
        
if __name__=='__main__':
    args = get_args()
    main(args)