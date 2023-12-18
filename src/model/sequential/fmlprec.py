import torch
import torch.nn as nn
from .seqrec_base import SeqRecBase
import src.model.fmlprec_modules as modules

class FMLPRec(SeqRecBase):
    def __init__(self, config):
        self.n_layers = config['n_layers']
        self.hidden_dropout_prob = config['hidden_dropout_prob']
        self.hidden_act = config['hidden_act']
        self.layer_norm_eps = float(config['layer_norm_eps'])
        self.max_seq_len = config['max_seq_len']
        super(FMLPRec, self).__init__(config)

    def _define_model_layers(self):
        # multi-head attention
        self.position_embedding = nn.Embedding(self.max_seq_len+1, self.hidden_size) # +1 for consistent with ranking model
        self.item_encoder = modules.Encoder(
            max_seq_len=self.max_seq_len,
            hidden_size=self.hidden_size,
            n_layers=self.n_layers,
            hidden_dropout_prob=self.hidden_dropout_prob,
            hidden_act=self.hidden_act
        )
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)

    def _get_attention_mask(self, item_seq):
        """Generate left-to-right uni-directional attention mask for multi-head attention."""
        attention_mask = (item_seq > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        # mask for left-to-right unidirectional
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long().to(item_seq.device)

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask


    def forward_user_emb(self, item_seq=None):
        is_adaretrieval = self.config.get('is_adaretrieval', 0)
        if not is_adaretrieval:
            predicted_user_emb =  self.forward_user_emb_batch(item_seq, [], [])
        else:
            reco_batch_cnt = self.config['reco_batch_cnt']
            reco_batch_size = self.config['reco_batch_size']

            predicted_user_emb = []
            predicted_topk_item_ids = []
            for cnt in range(reco_batch_cnt):
                cur_user_emb = self.forward_user_emb_batch(item_seq, predicted_user_emb, predicted_topk_item_ids)
                predicted_user_emb.append(cur_user_emb)
                cur_topk_item_ids = self.predict_topk_item_emb(cur_user_emb, reco_batch_size, predicted_topk_item_ids)
                predicted_topk_item_ids.append(cur_topk_item_ids)

        return predicted_user_emb
        

    def forward_user_emb_batch(self, item_seq=None, existing_user_emb=None, selected_item_ids=None):
        is_adaretrieval = self.config.get('is_adaretrieval', 0)
        position_ids = torch.arange(item_seq.size(1), dtype=torch.long, device=item_seq.device)
        position_ids = position_ids.unsqueeze(0).expand_as(item_seq)
        position_embedding = self.position_embedding(position_ids) #[batch_size, n_items_seq, embedding_size]
        item_emb = self.item_embedding_for_user(item_seq)
        if is_adaretrieval:
            # IRA
            item_emb = self.item_representation_adapter(item_emb, selected_item_ids)

        input_emb = item_emb + position_embedding #[batch_size, n_items_seq, embedding_size]
        input_emb = self.LayerNorm(input_emb)
        input_emb = self.dropout(input_emb)  
        extended_attention_mask = self._get_attention_mask(item_seq)

        item_encoded_layers = self.item_encoder(input_emb, extended_attention_mask,output_all_encoded_layers=True)
        user_emb = item_encoded_layers[-1]
        user_emb = user_emb[:, -1, :]
        if is_adaretrieval:
            # URA
            user_emb = self.user_representation_adapter(user_emb, existing_user_emb)

        return user_emb

