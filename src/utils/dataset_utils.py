'''
Author: 
Date: 2024-06-04 09:32:10
LastEditors: 
LastEditTime: 2024-06-04 09:36:39
Description: 
'''

def get_metric(dataset_name):
    """get_metric"""
    if dataset_name in ['moleculecla']:
        return 'rmse'
    else:
        raise ValueError(dataset_name)
    
def get_task_type(dataset_name):
    """get_metric"""
    if dataset_name in ['moleculecla']:
        return 'regression'
    else:
        raise ValueError(dataset_name)
    
def get_downstream_task_names(dataset_name):
    """
    Get task names of downstream dataset
    """
    if dataset_name =='moleculecla':
        return ['docking_score', 'glide_lipo', 'glide_hbond', 'glide_evdw', 'glide_ecoul', 'glide_erotb', 'glide_esite', 'glide_emodel', 'glide_einternal']
    else:
        raise ValueError('%s not supported' % dataset_name)