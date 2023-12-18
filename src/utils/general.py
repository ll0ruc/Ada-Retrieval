import time
from typing import List
import numpy as np
import torch
import random
import importlib
import os
import pandas as pd
from .file_io import *

def get_local_time_str():
    return time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    
def dict2str(d, sep=' '):
    a = [(k,v) for k,v in d.items()]
    a.sort(key=lambda t:t[0])
    res = sep.join(['{0}:{1}'.format(t[0], t[1]) for t in a])
    return res


def init_seed(seed):
    r""" init random seed for random functions in numpy, torch, cuda and cudnn

    Args:
        seed (int): random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True # could improve efficiency
    torch.backends.cudnn.deterministic = True # fix random seed


def get_class_instance(class_name, class_root='model'):
    r"""Automatically select class based on name

    Args:
        class_name (str): class name
        class_root (str): the root directory to search for the target class
        
    Returns:
        Recommender: a class
    """
    file_name = class_name.lower()
    module = None
    class_root = class_root.replace("/", ".")
    submodule_loc = os.path.dirname(importlib.import_module(class_root).__file__)
    for root, dirs, files in os.walk(submodule_loc, topdown=True):  
        module_path = root.replace(submodule_loc, class_root).replace('/', '.').replace('\\', '.') + '.' + file_name
        if importlib.util.find_spec(module_path):
            module = importlib.import_module(module_path)  
            break
    if module is None:
        err_msg = 'Cannot import `class name` [{0}] from {1}.'.format(class_name, class_root) 
        raise ValueError(err_msg)
    target_class = getattr(module, class_name)
    return target_class

r'''
    Load user history from file_name.
    Returns:
        User2History: An N-length ndarray (N is user count) of ndarray (of history items). 
                    E.g., User2History[user_id] is an ndarray of item history for user_id. 
'''
def load_user_history(file_path, file_name, n_users=None, format='user-item'):
    if os.path.exists(os.path.join(file_path, file_name + '.ftr')):
        df = pd.read_feather(os.path.join(file_path, file_name + '.ftr'))
    else:
        raise NotImplementedError("Unsupported user history file type: {0}".format(file_name) )

    if format == 'user-item':
        # ##TODO currently we only support one positive item
        # if isinstance(df['item_id'][0], list) or isinstance(df['item_id'][0], np.ndarray):
        #     df.loc[:, 'item_id'] = df.item_id.apply(lambda x: x[0])
        user_history = df.groupby('user_id')['item_id'].apply(lambda x:np.array(x))
    elif format == 'user-item_seq':
        # df['item_seq']=df['item_seq'].apply(lambda x:np.array(x))
        user_history = df.set_index('user_id')['item_seq' ].to_dict()
    else:
        raise NotImplementedError("Unsupport user history format: {0}".format(format))
    
    if n_users is None or n_users <= 0:
        n_users = df['user_id'].max() + 1
        print('Inferred n_users is {0}'.format(n_users))
    res = np.empty(n_users, dtype=object)
    for user_id, items in user_history.items():
        res[user_id] = items #np.fromiter(items, int, len(items))  

    return res
