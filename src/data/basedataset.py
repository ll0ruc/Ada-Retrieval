
import os
from datetime import datetime  
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import logging
import copy



class BaseDataset(Dataset):
    def __init__(self, config, path, filename, transform=None):
        self.config = config   
        self.logger = logging.getLogger(config['exp_name'])
        self.dataset_df = self.load_data(path, filename)
        ## remove unnessary data columns
        _valid_data_columns = self._get_valid_cols(self.dataset_df.columns.to_list())  
        self.dataset_df = self.dataset_df[_valid_data_columns]
        self.data_columns = self.dataset_df.columns

        ## When the format is 'user-item_seq':
        ## if it is training file, transform it to the user-item format
        if self.config['data_format'] in "user-item_seq" and self.config['data_loader_task'] == 'train':
            self.dataset_df = self.expand_dataset(self.dataset_df)
            self.config['data_format'] = 'user-item'

        self.dataset = self.dataset_df.values.astype(object)
        del self.dataset_df
        self.transform = transform
          
        self.set_return_column_index()
        self.logger.info('Finished initializing {0}'.format(self.__class__)) 

    def set_return_column_index(self): 
        _type = self.config['data_format']
        self.return_key_2_index = {
            'user_id':0,
            'item_id':1,
            'label':2  ## if the original data format does not contain label, it will append one fake label
        }

    def expand_dataset(self, dataset_df):
        res = dataset_df[['user_id', 'item_seq']] 
        res = res.explode('item_seq', ignore_index=True)
        res = res.rename(columns={'item_seq': 'item_id'}) 
        return res

    def _get_valid_cols(self, candidates):
        _type = self.config['data_format']
        if _type == 'user-item':
            t = ['user_id', 'item_id',]
        elif _type == 'user-item_seq':
            t = ['user_id', 'item_seq']
        else:
            raise ValueError(f"The file format `{_type}` is not supported now.")
        candidates = set(candidates)
        res = []
        for a in t:
            if a in candidates:
                res.append(a)
        return res

    r'''
    If there is no label column in the data file, will be appended with negative samples
    so only the first item in a group is the original postive item, the rest are all sampled negative.
    '''
    def _get_fake_label(self, items): 
        if hasattr(self, 'fake_label'):
            return self.fake_label
        if isinstance(items, list) or isinstance(items, np.ndarray):
            k = len(items)
            res = np.zeros((k,), dtype=np.int32)
            res[0] = 1 
        else:
            res = 1
        self.fake_label = res
        return res


    def __getitem__(self, index):
        _type = self.config['data_format']
        sample = self.dataset[index] 
        
        if self.transform is not None:
            sample = self.transform(sample)
        
        user_id = sample[0]
        item_id = sample[1] 
        
        if isinstance(item_id, list):
            item_id = np.asarray(item_id)
        if isinstance(item_id, np.ndarray):#to suppress the warning of pytorch: torch.as_tensor() in default_collate() will change the dataset itself
            item_id = copy.deepcopy(item_id)

        label = self._get_fake_label(item_id)
        if isinstance(label, list):
            label = np.asarray(label)

        return_tup = (user_id, item_id, label)
        return return_tup
         

    def __len__(self):
        return len(self.dataset)
 
    def load_basic_data(self, filename):
        self.logger.debug('loading {0} at {1}'.format(os.path.basename(filename),  datetime.now().strftime("%d/%m/%Y %H:%M:%S")))

        if os.path.exists(filename + '.ftr'):
            data = pd.read_feather(filename + '.ftr')
        else:
            raise NotImplementedError("Load plain text data file")

        self.logger.debug('Finished loading {0} at {1}'.format(os.path.basename(filename),  datetime.now().strftime("%d/%m/%Y %H:%M:%S")))
        data = data.reset_index(drop=True)
        return data
    
    def load_data(self, path, filename):
        filename = os.path.join(path, filename)
        data = self.load_basic_data(filename)  
        return data
