import os
import torch
from tqdm import tqdm 
import numpy as np
import inspect
from scipy import sparse as ssp

## all the supported evaluation metrics
SUPPORTED_RANKING_METRICS = {'ndcg', 'hit'}

## metrics that request the computation of ranks
METRICS_NEED_RANKS = {'ndcg', 'hit'}


def tensor2array(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    else:
        return data

class Evaluator(object): 
    def __init__(self, metrics_str=None, config=None, accelerator=None):
        self.config = config
        self.accelerator = accelerator
        self.NINF = -9999 # np.NINF
        self.metrics_need_ranks = METRICS_NEED_RANKS
    
    def evaluate_with_scores(self, scores, labels=None, **kwargs):
        raise NotImplementedError
    
    def _need_ranks(self, metrics_list):
        for metric in metrics_list:
            name = metric.split('@')[0] ## metric string is like ``name@k``
            if name in self.metrics_need_ranks:
                return True
        return False


    @torch.no_grad()
    def evaluate(self, data, model, verbose=0, predict_only=False):    
        model.eval()
        model = self.accelerator.unwrap_model(model)
        iter_data = (
            tqdm(
                enumerate(data),
                total=len(data),
                desc="Evaluate",
                dynamic_ncols=True,
                disable=not self.accelerator.is_local_main_process
            ) if verbose == 2 else enumerate(data)
        )
        
        all_scores = [] 
        all_labels, label_index = [], data.dataset.return_key_2_index['label']

        for _, inter_data in iter_data: 
            samples = {k:inter_data[v] for k,v in data.dataset.return_key_2_index.items()}
            scores = model.predict(samples)
            labels = inter_data[label_index]
            scores = self.accelerator.gather_for_metrics(torch.tensor(scores, device=self.accelerator.device)).cpu().numpy()
            labels = self.accelerator.gather_for_metrics(labels).cpu().numpy()
            all_scores.append(scores)
            all_labels.append(labels)

        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)

        if predict_only:
            return all_scores
        else:
            result = self.evaluate_with_scores(all_scores, all_labels)
            result = self.merge_scores(result)
            return result

    def remove_padding_items(self, batch_itemids):
        N = len(batch_itemids)
        res = np.empty(N, dtype=object)
        for i in range(N):
            res[i] = batch_itemids[i][batch_itemids[i] > 0]
        return res

    @torch.no_grad()
    def evaluate_with_full_items(self, data, model, user_history, verbose=0, infer_batch_size=40960):
        ## Attention: currently, only supports predict_layer_type=dot
        model.eval() 
        model = self.accelerator.unwrap_model(model)
        iter_data = (
            tqdm(
                enumerate(data),
                total=len(data),
                desc="Evaluate",
                dynamic_ncols=True,
                disable=not self.accelerator.is_local_main_process
            ) if verbose == 2 else enumerate(data)
        )
        
        all_results = [] 
        item_embeddings = model.forward_all_item_emb(infer_batch_size)
        if model.has_item_bias:
            item_bias = model.get_all_item_bias()
            item_bias = item_bias.reshape((1, -1))
        for idss, inter_data in iter_data:
            samples = {k:inter_data[v] for k,v in data.dataset.return_key_2_index.items()}
            inputs = {k: v for k, v in samples.items() if k in inspect.signature(model.forward_user_emb).parameters}
            user_embeddings = model.forward_user_emb(**inputs) 
            if isinstance(user_embeddings, list):
                adaretrieval = True
                tmp = []
                for _user_embeddings in user_embeddings:
                    tmp.append(tensor2array(_user_embeddings))
                user_embeddings = tmp
            else:
                adaretrieval = False
                user_embeddings = tensor2array(user_embeddings)           
            batch_userids = samples['user_id']
            batch_itemids = samples['item_id']
            batch_userids = tensor2array(batch_userids)
            batch_itemids = tensor2array(batch_itemids)
            if data.dataset.config['data_format'] == "user-item_seq":
                batch_itemids = self.remove_padding_items(batch_itemids)

            if isinstance(item_embeddings, ssp.spmatrix):
                # Therefore, a for-loop-based function decorated with numba.jit is used to accelerate.
                batch_scores = model.sparse_matrix_mul(user_embeddings, item_embeddings)
            else:
                if adaretrieval:
                    batch_scores = []
                    for _user_embeddings in user_embeddings:
                        batch_scores.append(_user_embeddings @ item_embeddings.T)
                else:
                    batch_scores = user_embeddings @ item_embeddings.T
            if isinstance(batch_scores, ssp.spmatrix):
                batch_scores = batch_scores.toarray()
            elif not adaretrieval:
                batch_scores = np.array(batch_scores)
                if model.has_item_bias:
                    batch_scores += item_bias

                batch_scores = batch_scores / self.config['tau']

                for idx, userid in enumerate(batch_userids):
                    itemid = batch_itemids[idx]
                    target_score = batch_scores[idx, itemid]

                    if userid < len(user_history) and user_history[userid] is not None:
                        history = user_history[userid]
                        batch_scores[idx][history] = self.NINF #Mask the item history
                    
                    if self.__class__.__name__ == 'OnePositiveEvaluator':
                        batch_scores[idx][0] = target_score
                        batch_scores[idx][itemid] = self.NINF
                    else:
                        batch_scores[idx][0] = self.NINF
                        batch_scores[idx][itemid] = target_score
                final_batch_scores = batch_scores

            else:
                # select the max top _k in slice.
                reco_batch_size = self.config['reco_batch_size']
                reco_batch_cnt = self.config['reco_batch_cnt']
                K = reco_batch_size * reco_batch_cnt
                final_batch_items = np.ones([len(batch_userids), K], dtype=np.int32)
                final_batch_scores_ = np.zeros([len(batch_userids), K])
                for pred_batch_cnt, singlie_slice_batch_scores in enumerate(batch_scores):
                    singlie_slice_batch_scores = np.array(singlie_slice_batch_scores)
                    if model.has_item_bias:
                        singlie_slice_batch_scores += item_bias
                    singlie_slice_batch_scores = singlie_slice_batch_scores / self.config['tau']

                    idxs = range(len(batch_userids))
                    singlie_slice_target_score = singlie_slice_batch_scores[idxs, batch_itemids]
                    for idx, userid in enumerate(batch_userids):
                        if userid < len(user_history) and user_history[userid] is not None:
                            history = user_history[userid]
                            singlie_slice_batch_scores[idx][history] = self.NINF
                    singlie_slice_batch_scores[idxs, batch_itemids] = singlie_slice_target_score
                    _k = reco_batch_size
                    ## mask previous selected items
                    if pred_batch_cnt > 0:
                        selected_items_len = _k * pred_batch_cnt
                        _previous_predictions = final_batch_items[:, :selected_items_len]
                        idxs = [[i] * selected_items_len for i in range(len(_previous_predictions))]
                        singlie_slice_batch_scores[idxs, _previous_predictions] = self.NINF

                    singlie_slice_batch_scores[:, 0] = self.NINF
                    singlie_slice_batch_scores = torch.from_numpy(singlie_slice_batch_scores).detach()
                    this_slice_predictions_score, this_slice_predictions_index = torch.topk(singlie_slice_batch_scores,
                                                                                            _k, dim=1, largest=True)
                    singlie_slice_batch_scores = singlie_slice_batch_scores.numpy()
                    this_slice_predictions_score = this_slice_predictions_score.numpy()
                    this_slice_predictions_index = this_slice_predictions_index.numpy()
                    final_batch_items[:,
                    pred_batch_cnt * _k:(pred_batch_cnt + 1) * _k] = this_slice_predictions_index
                    final_batch_scores_[:,
                    pred_batch_cnt * _k:(pred_batch_cnt + 1) * _k] = this_slice_predictions_score

                final_batch_scores = final_batch_items

            result = self.evaluate_with_scores(final_batch_scores, pos_itemids=batch_itemids)
            for k, v in result.items():
                result[k] = self.accelerator.gather_for_metrics(torch.tensor(v, device=self.accelerator.device)).cpu().numpy()
            all_results.append(result)
        result = self.merge_scores(all_results)
        return result

