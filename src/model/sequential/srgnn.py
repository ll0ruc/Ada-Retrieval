import torch
import torch.nn as nn
import numpy as np
from .seqrec_base import SeqRecBase
import src.model.modules as modules

class SRGNN(SeqRecBase):
    def __init__(self, config):
        self.step = config['step']
        super(SRGNN, self).__init__(config)

    def _define_model_layers(self):
        self.gnn = modules.GNN(self.hidden_size, self.step)
        self.linear_one = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_two = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_three = nn.Linear(self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

    def forward_user_emb(self, item_seq=None):
        is_adaretrieval = self.config.get('is_adaretrieval', 0)
        if not is_adaretrieval:
            predicted_user_emb =  self.forward_user_emb_batch(item_seq, [], [])
        else:
            reco_batch_cnt = self.config['reco_batch_cnt']
            reco_batch_size = self.config['reco_batch_size']
            predicted_user_emb = []
            predicted_topk_item_ids = []
            for _ in range(reco_batch_cnt):
                cur_user_emb = self.forward_user_emb_batch(item_seq, predicted_user_emb, predicted_topk_item_ids)
                predicted_user_emb.append(cur_user_emb)
                cur_topk_item_ids = self.predict_topk_item_emb(cur_user_emb, reco_batch_size, predicted_topk_item_ids)
                predicted_topk_item_ids.append(cur_topk_item_ids)

        return predicted_user_emb
        

    def forward_user_emb_batch(self, item_seq=None, existing_user_emb=None, selected_item_ids=None):
        is_adaretrieval = self.config.get('is_adaretrieval', 0)
        alias_inputs, A, items, mask = self._get_slice(item_seq)
        item_emb = self.item_embedding_for_user(items)
        if is_adaretrieval:
            # IRA
            item_emb = self.item_representation_adapter(item_emb, selected_item_ids)
        hidden = item_emb
        hidden = self.gnn(A, hidden)
        alias_inputs = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(
            -1, -1, self.embedding_size
        )
        seq_hidden = torch.gather(hidden, dim=1, index=alias_inputs)
        # fetch the last hidden state of last timestamp
        ht = seq_hidden[:, -1] # [B, D]
        q1 = self.linear_one(ht).view(ht.size(0), 1, ht.size(1))
        q2 = self.linear_two(seq_hidden)

        alpha = self.linear_three(torch.sigmoid(q1 + q2))
        a = torch.sum(alpha * seq_hidden * mask.view(mask.size(0), -1, 1).float(), 1)
        user_emb = self.linear_transform(torch.cat([a, ht], dim=1))

        if is_adaretrieval:
            # URA
            user_emb = self.user_representation_adapter(user_emb, existing_user_emb)

        return user_emb

    def _gather_indexes(self, output, gather_index):
        """Gathers the vectors at the specific positions over a minibatch"""
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])
        output_tensor = output.gather(dim=1, index=gather_index)
        return output_tensor.squeeze(1)

    def _get_slice(self, item_seq):
        # Mask matrix, shape of [batch_size, max_session_len]
        mask = item_seq.gt(0)
        items, n_node, A, alias_inputs = [], [], [], []
        max_n_node = item_seq.size(1)
        item_seq = item_seq.cpu().numpy()
        for u_input in item_seq:
            node = np.unique(u_input)
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            u_A = np.zeros((max_n_node, max_n_node))
            for i in np.arange(len(u_input) - 1):
                if u_input[i] == 0 or u_input[i+1] == 0:
                    continue
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                u_A[u][v] = 1

            u_sum_in = np.sum(u_A, 0)
            u_sum_in[np.where(u_sum_in == 0)] = 1
            u_A_in = np.divide(u_A, u_sum_in)
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out)
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            A.append(u_A)

            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
        # The relative coordinates of the item node, shape of [batch_size, max_session_len]
        alias_inputs = torch.LongTensor(alias_inputs).to(self.device)
        # The connecting matrix, shape of [batch_size, max_session_len, 2 * max_session_len]
        A = torch.FloatTensor(np.array(A)).to(self.device)
        # The unique item nodes, shape of [batch_size, max_session_len]
        items = torch.LongTensor(items).to(self.device)

        return alias_inputs, A, items, mask