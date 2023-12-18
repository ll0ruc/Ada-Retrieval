import numpy as np
import random


class AddUserHistory(object):
    r'''
    Parameters:
        user2history: An N-length ndarray (N is user count) of ndarray (of history items).
    '''
    def __init__(self, user2history, seq_last=0):
        self.user2history = user2history
        self.empty_history = np.zeros((1,), dtype=np.int32)
        self.seq_last = seq_last

    def get_fake_label(self, k):
        if hasattr(self, 'fake_label'):
            return self.fake_label  
        res = np.zeros((k,), dtype=np.int32)
        res[0] = 1  
        self.fake_label = res
        return res

    def __call__(self, sample): 
        ## suppose only one positive item;  
        ## sample is a ndarray of object : [userid, itemid, label, ...] 
        items = sample[1] 
        if isinstance(items, list) or isinstance(items, np.ndarray):
            items = set(items)
        else:
            items = set([items])
        userid = sample[0]
        if userid >= len(self.user2history) or self.user2history[userid] is None:
            history = self.empty_history
        else:
            history = self.user2history[userid]

        n = []
        for idx, item in enumerate(history):
            if item in items:
                n.append(idx)
        if len(n) == 0:
            pass
        else:
            n = n[-1] if self.seq_last else random.choice(n)
            history = history[:n]

        return history, len(history)
        