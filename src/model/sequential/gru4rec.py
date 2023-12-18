import torch.nn as nn

from .seqrec_base import SeqRecBase


class GRU4Rec(SeqRecBase):    
    def __init__(self, config):
        super(GRU4Rec, self).__init__(config)

    def _define_model_layers(self):
        # gru
        self.gru_hidden_size = self.config['gru_hidden_size']
        self.num_layers = self.config['n_layers']
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.gru_hidden_size,
            num_layers=self.num_layers,
            bias=True,
            batch_first=True,
        )
        self.dense = nn.Linear(self.gru_hidden_size, self.embedding_size)  

    def forward_user_emb(self, item_seq=None):
        is_adaretrieval = self.config.get('is_adaretrieval', 0)
        if not is_adaretrieval:
            predicted_user_emb = self.forward_user_emb_batch(item_seq, [], [])
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
        item_seq_emb = self.item_embedding_for_user(item_seq)
        is_adaretrieval = self.config.get('is_adaretrieval', 0)
        if is_adaretrieval:
            # IRA
            item_seq_emb = self.item_representation_adapter(item_seq_emb, selected_item_ids)

        item_seq_emb = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb)
        gru_output = self.dense(gru_output)

        user_emb = gru_output[:, -1]
        if is_adaretrieval:
            # URA
            user_emb = self.user_representation_adapter(user_emb, existing_user_emb)

        return user_emb


     