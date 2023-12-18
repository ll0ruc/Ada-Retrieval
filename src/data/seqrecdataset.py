
import numpy as np
from .basedataset import BaseDataset


class SeqRecDataset(BaseDataset):
    def __init__(self, config, path, filename, transform=None):
        super(SeqRecDataset, self).__init__(config, path, filename, transform) 
        self.add_seq_transform = None 

    def set_return_column_index(self):
        super(SeqRecDataset, self).set_return_column_index() 
        self.return_key_2_index['item_seq'] = len(self.return_key_2_index)
        self.return_key_2_index['item_seq_len'] = len(self.return_key_2_index)

    def add_user_history_transform(self, transform):
        self.add_seq_transform = transform

    def __getitem__(self, index):        
        elements = super(SeqRecDataset, self).__getitem__(index)   # user_id, item_id, label, ...
        item_seq, item_seq_len = self.add_seq_transform((elements[0], elements[1]))
        item_seq = self._padding(item_seq)
        item_seq_len = min(item_seq_len, self.config['max_seq_len'])
        elements = elements + (item_seq, item_seq_len)
        return elements

    def _padding(self, x): # padding item_seq to max_seq_len
        len_seq = len(x)
        k = self.config['max_seq_len']
        res = np.zeros((k,), dtype=np.int32)
        if len_seq < k:
            res[(k-len_seq):] = x[:]
        else:
            res[:] = x[len_seq-k:]
        return res
 