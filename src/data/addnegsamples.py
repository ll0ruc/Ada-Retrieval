import numpy as np
import random


class AddNegSamples(object):
    r"""
       Optional parameters:
            user2history: A N-len ndarray to store users' interacted items.
            item_popularity: A M-len ndarray to store items' popularity. 
    """
    def __init__(self, n_users, n_items, n_neg, **kwargs):
        self.n_users = n_users
        self.n_items = n_items
        self.n_neg = n_neg

        self.user2history_as_set = None
        self.neg_by_pop_alpha = 1.0

        for k,v in kwargs.items():
            if k == 'user2history' and v is not None:
                self.user2history_as_set = self._construct_history(v)

    r"""
        Convert ndarray to set for fast item checking.
        Parameters:
            user2history: A ndarray of ndarray.
        Returns:
            user2history_as_set: A ndarray of set.
    """
    def _construct_history(self, user2history):
        N = len(user2history)
        user2history_as_set = np.empty(N, dtype=object)
        for i in range(N):
            if user2history[i] is not None and len(user2history[i]) > 0:
                if isinstance(user2history[i], set):
                    user2history_as_set[i] = user2history[i]
                else:
                    user2history_as_set[i] = set(user2history[i])
        return user2history_as_set

    r"""
        Check the rationality of selected_item_id as a negative item.
    """
    def _valid(self, user_id, pos_item_id, selected_item_id):
        pos_set =  set(pos_item_id) if isinstance(pos_item_id, np.ndarray) else set([pos_item_id])
        if selected_item_id in pos_set:
            return False 
        if self.user2history_as_set is None or self.user2history_as_set[user_id] is None or selected_item_id not in self.user2history_as_set[user_id]:
            return True
        return False

    def _sample_one_item(self):

        return random.randint(1, self.n_items - 1)

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
        pos_item = sample[1] 
        pos_len = len(pos_item) if isinstance(pos_item, np.ndarray) else 1
        
        sampled_items = np.zeros(self.n_neg+pos_len, dtype=int) 
        for i in range(pos_len, self.n_neg+pos_len):
            retries = 100
            sampled_itemid = 0
            while retries > 0:
                idx = self._sample_one_item()
                if self._valid(sample[0], pos_item, idx):
                    sampled_itemid = idx
                    break
                retries -= 1
            sampled_items[i] = sampled_itemid

        sampled_items[0:pos_len] = pos_item 
        sample[1] = sampled_items
        if len(sample) >= 3: # I think if there exists label, it should not be replaced by fake label
            all_labels = np.zeros((self.n_neg+pos_len,), dtype=np.int32)
            all_labels[0:pos_len] = sample[2]
            sample[2] = all_labels
        return sample
    
        