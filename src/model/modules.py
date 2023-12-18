import torch
import torch.nn as nn
import copy
import math
from typing import *
from torch.nn import functional as F

EPS = 1e-8


class InnerProductScorer(nn.Module):
    def __init__(self):
        super(InnerProductScorer, self).__init__()

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        assert math.fabs(x.dim() - y.dim()) <= 1, "difference of dimension between x and y should be no more than 1."
        # 3 cases. Some cases could be merged but not easy to understanding after merge
        ## case1: x.dim() == y.dim(), e.g. [N,D]x[N,D]-one user vs one item, [N,D]x[M,D]-each user vs all items,  ([B,N,D]x[B,N,D], [B,N,D]x[B,M,D], optional, not needed now)
        if x.dim() == y.dim():
            if x.size(0) == y.size(0):  # one user vs one item
                res = (x * y).sum(-1)   # [N,]
            else:   # each user vs all items
                res = torch.matmul(x, y.transpose(-1,-2))   # [N, M]
        ## case2: x.dim() = y.dim()+1, e.g. [N,D]x[D], [B,N,D]x[B,D] , one item vs multiple users
        elif x.dim() > y.dim():
            if y.dim() == 1:
                res = torch.matmul(x, y)    #[N,]
            else:
                res = torch.matmul(x, y.view(*y.shape, 1)).squeeze(-1)   #[B, N]
        ## case3: x.dim() = y.dim()-1, e.g. [D]x[N,D], [B,D]x[B,N,D], one user vs multiple items
        else:
            res = self.forward(y, x)
        return res

# Transformer
class MultiHeadAttention(nn.Module):
    """
    Multi-head Self-attention layers, a attention score dropout layer is introduced.
    Args:
        input_tensor (torch.Tensor): the input of the multi-head self-attention layer
        attention_mask (torch.Tensor): the attention mask for input tensor
    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer
    """

    def __init__(self, n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps):
        super(MultiHeadAttention, self).__init__()
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads
        self.attention_head_size = int(hidden_size / n_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class FeedForward(nn.Module):
    """
    Point-wise feed-forward layer is implemented by two dense layers.
    Args:
        input_tensor (torch.Tensor): the input of the point-wise feed-forward layer
    Returns:
        hidden_states (torch.Tensor): the output of the point-wise feed-forward layer
    """

    def __init__(self, hidden_size, inner_size, hidden_dropout_prob, hidden_act, layer_norm_eps):
        super(FeedForward, self).__init__()
        self.dense_1 = nn.Linear(hidden_size, inner_size)
        self.intermediate_act_fn = self.get_hidden_act(hidden_act)

        self.dense_2 = nn.Linear(inner_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def get_hidden_act(self, act):
        ACT2FN = {
            "gelu": nn.GELU(approximate='none'), # approximate can be set to 'tanh' for OpenAI GPT's gelu: 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
            "relu": nn.ReLU(),
            "swish": nn.SiLU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid(),
        }
        return ACT2FN[act]

    def forward(self, input_tensor):
        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class TransformerLayer(nn.Module):
    """
    One transformer layer consists of a multi-head self-attention layer and a point-wise feed-forward layer.
    Args:
        hidden_states (torch.Tensor): the input of the multi-head self-attention sublayer
        attention_mask (torch.Tensor): the attention mask for the multi-head self-attention sublayer
    Returns:
        feedforward_output (torch.Tensor): The output of the point-wise feed-forward sublayer,
                                           is the output of the transformer layer.
    """

    def __init__(
        self, n_heads, hidden_size, intermediate_size, hidden_dropout_prob, attn_dropout_prob, hidden_act,
        layer_norm_eps
    ):
        super(TransformerLayer, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            n_heads, hidden_size, hidden_dropout_prob, attn_dropout_prob, layer_norm_eps
        )
        self.feed_forward = FeedForward(hidden_size, intermediate_size, hidden_dropout_prob, hidden_act, layer_norm_eps)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.multi_head_attention(hidden_states, attention_mask)
        feedforward_output = self.feed_forward(attention_output)
        return feedforward_output


class TransformerEncoder(nn.Module):
    r""" One TransformerEncoder consists of several TransformerLayers.
        - n_layers(num): num of transformer layers in transformer encoder. Default: 2
        - n_heads(num): num of attention heads for multi-head attention layer. Default: 2
        - hidden_size(num): the input and output hidden size. Default: 64
        - inner_size(num): the dimensionality in feed-forward layer. Default: 256
        - hidden_dropout_prob(float): probability of an element to be zeroed. Default: 0.5
        - attn_dropout_prob(float): probability of an attention score to be zeroed. Default: 0.5
        - hidden_act(str): activation function in feed-forward layer. Default: 'gelu'
                      candidates: 'gelu', 'relu', 'swish', 'tanh', 'sigmoid'
        - layer_norm_eps(float): a value added to the denominator for numerical stability. Default: 1e-12
    """

    def __init__(
        self,
        n_layers=2,
        n_heads=2,
        hidden_size=64,
        inner_size=256,
        hidden_dropout_prob=0.5,
        attn_dropout_prob=0.5,
        hidden_act='gelu',
        layer_norm_eps=1e-12
    ):

        super(TransformerEncoder, self).__init__()
        layer = TransformerLayer(
            n_heads, hidden_size, inner_size, hidden_dropout_prob, attn_dropout_prob, hidden_act, layer_norm_eps
        )
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        """
        Args:
            hidden_states (torch.Tensor): the input of the TransformerEncoder
            attention_mask (torch.Tensor): the attention mask for the input hidden_states
            output_all_encoded_layers (Bool): whether output all transformer layers' output
        Returns:
            all_encoder_layers (list): if output_all_encoded_layers is True, return a list consists of all transformer
            layers' output, otherwise return a list only consists of the output of last transformer layer.
        """
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class GNN(nn.Module):
    r"""Graph neural networks are well-suited for session-based recommendation,
    because it can automatically extract features of session graphs with considerations of rich node connections.
    """

    def __init__(self, embedding_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.embedding_size = embedding_size
        self.input_size = embedding_size * 2
        self.gate_size = embedding_size * 3
        self.w_ih = nn.Parameter(torch.randn(self.gate_size, self.input_size))
        self.w_hh = nn.Parameter(torch.randn(self.gate_size, self.embedding_size))
        self.b_ih = nn.Parameter(torch.randn(self.gate_size))
        self.b_hh = nn.Parameter(torch.randn(self.gate_size))
        self.b_iah = nn.Parameter(torch.randn(self.embedding_size))
        self.b_ioh = nn.Parameter(torch.randn(self.embedding_size))

        self.linear_edge_in = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
        self.linear_edge_out = nn.Linear(self.embedding_size, self.embedding_size, bias=True)

    def GNNCell(self, A, hidden):
        r"""Obtain latent vectors of nodes via graph neural networks.
        Args:
            A(torch.FloatTensor):The connection matrix,shape of [batch_size, max_session_len, 2 * max_session_len]
            hidden(torch.FloatTensor):The item node embedding matrix, shape of
                [batch_size, max_session_len, embedding_size]
        Returns:
            torch.FloatTensor: Latent vectors of nodes,shape of [batch_size, max_session_len, embedding_size]
        """

        input_in = torch.matmul(A[:, :, :A.size(1)], self.linear_edge_in(hidden)) + self.b_iah
        input_out = torch.matmul(A[:, :, A.size(1):2 * A.size(1)], self.linear_edge_out(hidden)) + self.b_ioh
        # [batch_size, max_session_len, embedding_size * 2]
        inputs = torch.cat([input_in, input_out], 2)

        # gi.size equals to gh.size, shape of [batch_size, max_session_len, embedding_size * 3]
        gi = F.linear(inputs, self.w_ih, self.b_ih)
        gh = F.linear(hidden, self.w_hh, self.b_hh)
        # (batch_size, max_session_len, embedding_size)
        i_r, i_i, i_n = gi.chunk(3, 2)
        h_r, h_i, h_n = gh.chunk(3, 2)
        reset_gate = torch.sigmoid(i_r + h_r)
        input_gate = torch.sigmoid(i_i + h_i)
        new_gate = torch.tanh(i_n + reset_gate * h_n)
        hy = (1 - input_gate) * hidden + input_gate * new_gate
        return hy

    def forward(self, A, hidden):
        for i in range(self.step):
            hidden = self.GNNCell(A, hidden)
        return hidden

class ResidualBlock_b(nn.Module):
    r"""
    Residual block (b) in the paper
    """

    def __init__(self, in_channel, out_channel, kernel_size=3, dilation=None):
        super(ResidualBlock_b, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation)
        self.ln1 = nn.LayerNorm(out_channel, eps=1e-8)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation * 2)
        self.ln2 = nn.LayerNorm(out_channel, eps=1e-8)

        self.dilation = dilation
        self.kernel_size = kernel_size

    def forward(self, x):  # x: [batch_size, seq_len, embed_size]
        x_pad = self.conv_pad(x, self.dilation)  # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*dilations]
        out = self.conv1(x_pad).squeeze(2).permute(0, 2, 1)
        # [batch_size, seq_len+(self.kernel_size-1)*dilations-kernel_size+1, embed_size]
        out = F.relu(self.ln1(out))
        out_pad = self.conv_pad(out, self.dilation * 2)
        out2 = self.conv2(out_pad).squeeze(2).permute(0, 2, 1)
        out2 = F.relu(self.ln2(out2))
        return out2 + x

    def conv_pad(self, x, dilation):
        r""" Dropout-mask: To avoid the future information leakage problem, this paper proposed a masking-based dropout
        trick for the 1D dilated convolution to prevent the network from seeing the future items.
        Also the One-dimensional transformation is completed in this function.
        """
        inputs_pad = x.permute(0, 2, 1)
        inputs_pad = inputs_pad.unsqueeze(2)
        pad = nn.ZeroPad2d(((self.kernel_size - 1) * dilation, 0, 0, 0))
        inputs_pad = pad(inputs_pad)
        return inputs_pad