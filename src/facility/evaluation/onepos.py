
import numba
from src.facility.evaluation.evaluator_abc import *

@numba.jit(nopython=True)
def _get_ndcg_weights(length):
    ndcg_weights = 1 / np.log2(np.arange(2, length + 2))
    return ndcg_weights

@numba.jit(nopython=True)
def _get_mrr_weights(length):
    mrr_weights = 1 / np.arange(1, length + 1)
    return mrr_weights

@numba.jit(nopython=True, parallel=True)
def get_rank(A):
    rank = np.empty(len(A), dtype=np.int32)
    for i in numba.prange(len(A)):
        a = A[i]
        key = a[0]
        r = 0
        for j in range(1, len(a)):
            if a[j] > key:
                r += 1
        rank[i] = r
    return rank

@numba.jit(nopython=True, parallel=True)
def search_pos_id(A, L):
    rank = np.empty(len(A), dtype=np.int32)
    for i in numba.prange(len(A)):
        a = A[i]
        key = L[i]
        rank[i] = len(a)
        for j in range(0, len(a)):
            if a[j] == key:
                rank[i] = j
                break
    return rank


class OnePositiveEvaluator(Evaluator):    
    def __init__(self, metrics_str=None, config=None, accelerator=None):
        super(OnePositiveEvaluator, self).__init__(metrics_str, config, accelerator)
        self.metrics_list = eval(metrics_str)
        self.noise = {}
        self.zero_vec = {}
        self.zero_vec_mask_k = {}
        
    def get_zero_vec(self, k):
        if k not in self.zero_vec:
            self.zero_vec[k] = np.zeros((k,), dtype=np.float32)
        return self.zero_vec[k]

    def get_zero_vec_mask_k(self, n, k):
        if (n,k) not in self.zero_vec_mask_k:
            if k == np.Inf:
                self.zero_vec_mask_k[(n,k)] = np.ones((n,), dtype=np.float32)  
            else:
                self.zero_vec_mask_k[(n,k)] = np.zeros((n,), dtype=np.float32)  
                self.zero_vec_mask_k[(n,k)][:k] = 1.0
            
        return self.zero_vec_mask_k[(n,k)]

    def ndcg(self, k, rank, w):    
        masker = self.get_zero_vec_mask_k(len(w), k)
        res = w[rank] * masker[rank] 
        return res
    
    def hit(self, k, rank):
        top_items = rank < k
        return top_items + 0.0 ## convert bool to float

    def evaluate_with_scores(self, scores, labels=None, **kwargs):
        is_adaretrieval = self.config.get('is_adaretrieval', 0)
        if is_adaretrieval:
            _k = self.config['reco_batch_size']
            c = self.config['reco_batch_cnt']
            S = scores
            pos_itemids = kwargs['pos_itemids']
            num_scores = S.shape[1]+1
            rank = search_pos_id(S, pos_itemids)
            ndcg_w = _get_ndcg_weights(num_scores)
            res = {}
            K = _k * c
            for metric in self.metrics_list:
                tokens = metric.split('@')
                key, ks = tokens[0], tokens[1].split(';')
                if key == 'ndcg':
                    for k in ks:
                        if int(k) > K:
                            continue
                        res['{0}@{1}'.format(key, k)] = self.ndcg(int(k), rank, ndcg_w)
                elif key == 'hit':
                    for k in ks:
                        if int(k) > K:
                            continue
                        res['{0}@{1}'.format(key, k)] = self.hit(int(k), rank)
                else:
                    raise ValueError('metric {0} is unknown.'.format(key))

        else:
            S = scores
            res = {}
            num_scores = S.shape[1]+1

            # add small perturbation
            shape_key = S.shape
            if shape_key not in self.noise:
                self.noise[shape_key] = np.random.uniform(low=-1e-8, high=1e-8, size=S.shape)

            S += self.noise[shape_key]
            rank = get_rank(S)
            ndcg_w = _get_ndcg_weights(num_scores)

            for metric in self.metrics_list:
                if '@' in metric:
                    tokens = metric.split('@')
                    key, ks = tokens[0], tokens[1].split(';')
                    if key == 'ndcg':
                        for k in ks:
                            res['{0}@{1}'.format(key, k)] = self.ndcg(int(k), rank, ndcg_w)
                    elif key == 'hit':
                        for k in ks:
                            res['{0}@{1}'.format(key, k)] = self.hit(int(k), rank)

                    else:
                        raise ValueError('metric {0} is unknown.'.format(key))

        return res

    def merge_scores(self, all_results):
        overall_scores =  self.merge_scores_core(all_results)
        all_res = overall_scores
        return all_res

    def merge_scores_core(self, all_results):   
        if isinstance(all_results, list):
            if len(all_results) > 0:
                res = {}
                keys = all_results[0].keys()
                for metric_key in keys:
                    res[metric_key] = np.concatenate([t[metric_key] for t in all_results])
            else: 
                res = all_results[0]   
        else: 
            res = all_results
        
        keys = res.keys()
         
        all_res = {} 
        for key in keys:
            if not key.startswith('_'): # not temporary record
                all_res[key] = res[key].mean()

        return all_res
