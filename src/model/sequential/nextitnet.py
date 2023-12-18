import torch
import torch.nn as nn
from .seqrec_base import SeqRecBase
import src.model.modules as modules

class NextItNet(SeqRecBase):
    def __init__(self, config):
        self.block_num = config['block_num']
        self.dilations = [1,4] * self.block_num
        self.kernel_size = config['kernel_size']
        super(NextItNet, self).__init__(config)

    def _define_model_layers(self):
        # residual blocks    dilations in blocks:[1,2,4,8,1,2,4,8,...]
        self.residual_channels = self.hidden_size
        rb = [modules.ResidualBlock_b(
             in_channel=self.residual_channels,
             out_channel=self.residual_channels,
             kernel_size=self.kernel_size,
             dilation=dilation
            ) for dilation in self.dilations
        ]
        self.residual_blocks = nn.Sequential(*rb)
        # fully-connected layer
        self.final_layer = nn.Linear(self.residual_channels, self.embedding_size)

    
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
        item_emb = self.item_embedding_for_user(item_seq)
        if is_adaretrieval:
            # IRA
            item_emb = self.item_representation_adapter(item_emb, selected_item_ids)

        # Residual locks
        dilate_outputs = self.residual_blocks(item_emb)
        hidden = dilate_outputs[:, -1, :].view(-1, self.residual_channels)  # [batch_size, embed_size]
        user_emb = self.final_layer(hidden)  # [batch_size, embedding_size]

        if is_adaretrieval:
            # URA
            user_emb = self.user_representation_adapter(user_emb, existing_user_emb)

        return user_emb
