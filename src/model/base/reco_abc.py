import math
import numpy as np
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_
import src.model.fmlprec_modules as filterlayer

def xavier_normal_initialization(module):
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)
        if module.padding_idx is not None:
            constant_(module.weight.data[module.padding_idx], 0.)
    elif isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

def normal_initialization(mean, std):
    def normal_init(module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=mean, std=std)
            if module.padding_idx is not None:
                constant_(module.weight.data[module.padding_idx], 0.)
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=mean, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    return normal_init

def xavier_uniform_initialization(module):
    if isinstance(module, nn.Embedding):
        xavier_uniform_(module.weight.data)
        if module.padding_idx is not None:
            constant_(module.weight.data[module.padding_idx], 0.)
    elif isinstance(module, nn.Linear):
        xavier_uniform_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)

class AbstractRecommender(nn.Module):    
    def __init__(self, config):
        super(AbstractRecommender, self).__init__()
        self.logger = logging.getLogger(config['exp_name'])    
        self.__optimized_by_SGD__ = True
        self.config = config        
        self._init_attributes()
        self._init_modules() 
        
        self.annotatins = []
        self.add_annotation()



    def _define_model_layers(self):
        raise NotImplementedError

    ## -------------------------------
    ## Ada-Retrieval functions you need to pay attention to.
    ## -------------------------------
    def _define_ada_model_layers(self):
        ## modules for ada-retriever
        self.hidden_dropout_prob = self.config['hidden_dropout_prob']
        self.reco_batch_size = self.config['reco_batch_size']
        self.hidden_act = self.config['hidden_act']
        self.low_filter = filterlayer.LowFilterLayer(
            max_seq_len=self.reco_batch_size,
            hidden_size=self.hidden_size,
            hidden_dropout_prob=self.hidden_dropout_prob
        )
        self.att_dropout = nn.Dropout(self.hidden_dropout_prob)
        self.att_LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.post_trans_gru_layer = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.embedding_size,
            num_layers=1,
            bias=True,
            batch_first=True,
        )
        self.ffn_dense_1 = nn.Linear(self.hidden_size * 2, self.hidden_size * 2)
        self.relu = nn.ReLU()
        self.ffn_dense_2 = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.ffn_dropout = nn.Dropout(self.hidden_dropout_prob)
        self.ffn_LayerNorm = nn.LayerNorm(self.hidden_size, eps=1e-12)

    def item_representation_adapter(self, input_emb, selected_item_ids):
        res = input_emb
        if selected_item_ids is not None and len(selected_item_ids) > 0:
            selected_item_emb = []
            for item_ids in selected_item_ids:
                s_item_emb = self.item_embedding(item_ids)  # [B,T,D]
                s_item_emb_ = self.low_filter(s_item_emb)
                selected_item_emb.append(s_item_emb_)
            selected_item_emb = torch.cat(selected_item_emb, dim=1)
            ## context-aware attention
            sqrt_attention_size = math.sqrt(self.hidden_size)
            attention_scores = torch.bmm(input_emb, selected_item_emb.transpose(1, 2)) / sqrt_attention_size
            attention_weights = F.softmax(attention_scores, dim=2)
            context_item_emb = torch.bmm(attention_weights, selected_item_emb)
            hidden_states = self.att_dropout(context_item_emb)
            res = self.att_LayerNorm(hidden_states + input_emb)
        return res

    def user_representation_adapter(self, user_emb, existing_user_emb):
        res = user_emb
        if existing_user_emb is not None and len(existing_user_emb) > 0:
            existing_user_emb = torch.stack(existing_user_emb, dim=1)
            context_emb_, _ = self.post_trans_gru_layer(existing_user_emb)
            context_emb = context_emb_[:, -1]
            ## MLP layer
            hidden_states = self.ffn_dense_1(torch.cat([user_emb, context_emb], dim=-1))
            hidden_states = self.relu(hidden_states)
            hidden_states = self.ffn_dense_2(hidden_states)
            hidden_states = self.ffn_dropout(hidden_states)
            res = self.ffn_LayerNorm(hidden_states + user_emb)
        return res

    def forward(self, user_id):
        raise NotImplementedError

    def forward_user_emb(self, interaction):
        raise NotImplementedError

    def forward_item_emb(self, interaction):
        raise NotImplementedError

    ## -------------------------------
    ## More functions you may need to override.
    ## -------------------------------
    def _predict_layer(self, user_emb, items_emb, interaction):  
        raise NotImplementedError  
    
    def predict(self, interaction):
        raise NotImplementedError  
    
    def add_annotation(self):
        self.annotatins.append('AbstractRecommender')


    ## -------------------------------
    ## Belowing functions can fit most scenarios so you don't need to override.
    ## ------------------------------- 
    
    def _init_attributes(self):
        config = self.config
        self.n_users = config['n_users']
        self.n_items = config['n_items']
        self.device = config['device']
        self.embedding_size = config.get('embedding_size', 0)
        self.hidden_size = self.embedding_size
        self.dropout_prob = config.get('dropout_prob', 0.0)
        self.use_pre_item_emb = config.get('use_pre_item_emb', 0)
        self.init_method = config.get('init_method', 'normal')

        ## clip the score to avoid loss being nan
        ## usually this is not necessary, so you don't need to set up score_clip_value in config
        self.SCORE_CLIP = -1
        if 'score_clip_value' in self.config:
            self.SCORE_CLIP = self.config['score_clip_value']
        self.has_item_bias = False
        if 'has_item_bias' in config:
            self.has_item_bias = config['has_item_bias']
        self.tau = config.get('tau', 1.0)

    def _init_modules(self):
        # define layers and loss
        # TODO: remove user_embedding when user_id is not needed to save memory. Like in VAE.
        if self.has_item_bias:
            self.item_bias = nn.Parameter(torch.normal(0, 0.1, size=(self.n_items,)))
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size, padding_idx=0) #if padding_idx is not set, the embedding vector of padding_idx will change during training
        self._define_model_layers()
        base_params = set(self._modules.keys())
        self._init_params()
        self.base_params = base_params
        is_adaretrieval = self.config.get('is_adaretrieval', 0)
        if is_adaretrieval:
            self._define_ada_model_layers()
            extra_paras = set(self._modules.keys()) - base_params
            self._init_params(extra_paras)


    def _init_params(self, paras_list=None):
        init_methods = {
            'xavier_normal': xavier_normal_initialization,
            'xavier_uniform': xavier_uniform_initialization,
            'normal': normal_initialization(self.config['init_mean'], self.config['init_std']),
        }
        for name, module in self.named_children():
            if paras_list is not None:
                if name not in paras_list:
                    continue
            init_method = init_methods[self.init_method]
            module.apply(init_method)
    
    def _cal_loss(self, scores, labels=None, reduction=True):

        logits = torch.clamp(nn.Sigmoid()(scores), min=-1*1e-8, max=1-1e-8)
        labels = labels.float()#.to(self.device)
        loss = nn.BCELoss(reduction='mean' if reduction else 'none')(logits, labels).mean(dim=-1)

        return loss   

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        
        messages = []
        messages.append(super().__str__())
        messages.append('Trainable parameter number: {0}'.format(params))
        
        messages.append('All trainable parameters:')
        for name, param in self.named_parameters():
            if param.requires_grad:
                messages.append('{0} : {1}'.format(name, param.size()))
        
        return '\n'.join(messages)

