import torch.nn as nn
import torch.nn.functional as F
import torch

import math
import numpy as np

class Linear(nn.Module):
    '''
    create linear combinations for different embedded vectors
    '''
    def __init__(self, dropout):
        super(Linear, self).__init__()
#         self.weights = torch.nn.Parameter(F.softmax(torch.rand(ncov), dim=-1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x_emb):
        # with shape [batch_size, ncov, m]
#         x_emb = x_emb.permute(0,2,1)
        # now it has shape [batch_size, m, ncov]
        # comparing to an image, m is like the channel
        return self.flatten(self.dropout(x_emb))
    
class LinearAtt(nn.Module):
    '''
    create linear combinations for different embedded vectors
    '''
    def __init__(self, ncov, dropout):
        super(LinearAtt, self).__init__()
#         self.weights = torch.nn.Parameter(F.softmax(torch.rand(ncov), dim=-1))
#         self.flatten = nn.Flatten()
#         self.dropout = nn.Dropout(p=dropout)
        self.weights = torch.nn.Parameter(F.softmax(torch.rand(ncov), dim=-1))

    def forward(self, x_emb):
        # with shape [batch_size, ncov, m]
        x_emb = x_emb.permute(0,2,1)
        # now it has shape [batch_size, m, ncov]
        # comparing to an image, m is like the channel
        return (x_emb * self.weights).sum(-1)
        
        
        
class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def __init__(self, dropout=0.1):
        super(Attention, self).__init__()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, query, key, value, mask=None, flatten=True):

        # scores = torch.matmul(query, key.transpose(-2, -1)) \
        #          / math.sqrt(query.size(-1))
        
        # f(query, key)
        # query: d \times m
        # (d x m ) x (m x d)
        # d \times d
        device = value.device
        scores = torch.exp(torch.matmul(query, key.transpose(-2, -1))) \
                 / math.sqrt(query.size(-1))
        del key, query

        if mask is not None:
            # mask missing points before softmax
            scores = scores.masked_fill(mask.to(device) == 0, -1e9)
            del mask

        p_attn = F.softmax(scores, dim=-1)
        del scores

#         if dropout is not None:
        p_attn = self.dropout(p_attn)
        output = torch.matmul(p_attn, value)
        
        if flatten:
#             return self.flatten(output), p_attn
            return output.sum(1), p_attn
        else:
            return output, p_attn

class SelfAttention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """
    def __init__(self, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x, mask=None, flatten=True):

        # scores = torch.matmul(query, key.transpose(-2, -1)) \
        #          / math.sqrt(query.size(-1))
        
        # f(query, key)
        # query: d \times m
        # (d x m ) x (m x d)
        # d \times d
        device = x.device
        scores = torch.exp(torch.matmul(x, x.transpose(-2, -1))) \
                 / math.sqrt(x.size(-1))

        if mask is not None:
            # mask missing points before softmax
            scores = scores.masked_fill(mask.to(device) == 0, -1e9)
            del mask

        p_attn = F.softmax(scores, dim=-1)
        del scores

#         if dropout is not None:
        p_attn = self.dropout(p_attn)
        output = torch.matmul(p_attn, x)
        
        if flatten:
#             return self.flatten(output), p_attn
            return output.sum(1), p_attn
        else:
            return output, p_attn    
    
class MultiHeadedAttention(nn.Module):
    """
    Take in models size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model, bias=True) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model, bias=True)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # the same mask applies to all heads
            # unsqueeze Returns a new tensor with a dimension of size one
            # inserted at the specified position.
            mask = mask.unsqueeze(1)

        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention.forward(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)
    