import numpy as np
import inspect
import torch
import math
import src.model.modules as modules
from .reco_abc import AbstractRecommender

class BaseRecommender(AbstractRecommender): 
        
    def _init_attributes(self):
        super(BaseRecommender, self)._init_attributes()
        config = self.config
        self.dnn_inner_size = self.embedding_size
        self.time_seq = config.get('time_seq', 0)
    
    def _init_modules(self):
        # predict_layer
        self.scorer_layers = modules.InnerProductScorer().to(self.device)
        super(BaseRecommender, self)._init_modules()

    def _define_model_layers(self):
        pass

    def forward_user_emb(self, item_seq=None):
        pass
    
    def forward(self, user_id=None, item_id=None, label=None, item_seq=None, item_seq_len=None, reduction=True, return_loss_only=True):

        in_item_id = item_id
        items_emb = self.forward_item_emb(in_item_id) #[batch_size, n_in_item, embedding_size]
        user_emb = self.forward_user_emb(item_seq) #[batch_size, embedding_size]
        if isinstance(user_emb, list):
            scores = []
            for single_slice_user_emb in user_emb:
                scores.append(self._predict_layer(single_slice_user_emb, items_emb, in_item_id))
        else:
            scores = self._predict_layer(user_emb, items_emb, in_item_id)
        if self.training:
            if isinstance(scores, list):
                loss = 0
                _batch = 0
                for single_slice_scores in scores:
                    lam = self.config['lambda']
                    loss += self._cal_loss(single_slice_scores, label, reduction)  * math.pow(lam, _batch)
                    _batch += 1
            else:
                loss = self._cal_loss(scores, label, reduction)
            if return_loss_only:
                return loss, None, None, None
            return loss, scores, user_emb, items_emb
        else:
            return None, scores, user_emb, items_emb
    
    def forward_item_emb(self, items):
        item_emb = self.item_embedding(items) # [batch_size, n_items_inline, embedding_size]
        return item_emb

    def _predict_layer(self, user_emb, items_emb, item_id):
        scores = self.scorer_layers(user_emb, items_emb)
        if self.has_item_bias:
            item_bias = self.item_bias[item_id]
            scores = scores + item_bias

        scores = scores / self.tau

        if self.SCORE_CLIP > 0:
            scores = torch.clamp(scores, min=-1.0*self.SCORE_CLIP, max=self.SCORE_CLIP) 
        return scores  


    def predict(self, interaction):
        items_emb = self.forward_item_emb(interaction['item_id'])
        inputs = {k: v for k, v in interaction.items() if k in inspect.signature(self.forward_user_emb).parameters}
        user_emb = self.forward_user_emb(**inputs)
        item_id = interaction['item_id'] if 'item_id' in interaction else None
        scores = self._predict_layer(user_emb, items_emb, item_id).detach().cpu().numpy()
        return scores

    def forward_all_item_emb(self, batch_size=None, numpy=True):
        ### get all item's embeddings. when batch_size=None, it will proceed all in one run. 
        ### when numpy=False, it would return a torch.Tensor
        if numpy:
            res = np.zeros((self.n_items, self.embedding_size), dtype=np.float32)
        else:
            res = torch.zeros((self.n_items, self.embedding_size), dtype=torch.float32, device=self.device)
        if batch_size is None:
            batch_size = self.n_items
        
        n_batch = (self.n_items - 1) // batch_size + 1
        for batch in range(n_batch):
            start = batch * batch_size
            end = min(self.n_items, start + batch_size)
            cur_items = torch.arange(start, end, dtype=torch.int32, device=self.device)
            cur_items_emb = self.forward_item_emb(cur_items).detach()
            if numpy:
                cur_items_emb = cur_items_emb.cpu().numpy()
            res[start:end] = cur_items_emb
        return res    

    def get_all_item_bias(self):
        return self.item_bias.detach().cpu().numpy()        

    def item_embedding_for_user(self, item_seq):
        item_emb = self.item_embedding(item_seq)
        return item_emb

    def predict_topk_item_emb(self, user_emb, k, item_hist):
        all_item_emb = self.forward_all_item_emb(numpy=False)
        all_items_id = torch.arange(0, self.n_items, dtype=torch.long, device=self.device)
        all_scores = self._predict_layer(user_emb, all_item_emb, all_items_id)
        all_scores[:, 0] = - torch.inf
        if item_hist is not None and len(item_hist) > 0:
            item_hist = torch.cat(item_hist, dim=-1)
            row_idx = torch.arange(user_emb.size(0), dtype=torch.long).unsqueeze_(-1).expand_as(item_hist)
            all_scores[row_idx, item_hist] = - torch.inf
        _, topk_ids = torch.topk(all_scores, k, dim=-1)
        res = topk_ids
        return res