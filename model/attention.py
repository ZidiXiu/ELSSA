import torch.nn as nn
import torch.nn.functional as F
import torch

import math
import numpy as np
from utils.trainer_helpers import attention_mask

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
    def __init__(self, dropout=0.1, alpha = 0.9):
        super(SelfAttention, self).__init__()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout)
        self.alpha = alpha
    def forward(self, x, mask=None, flatten=True, aggregate=False, recompute=True):

        # scores = torch.matmul(query, key.transpose(-2, -1)) \
        #          / math.sqrt(query.size(-1))
        
        # f(query, key)
        # query: d \times m
        # (d x m ) x (m x d)
        # d \times d
        device = x.device
        if not aggregate:
            recompute = True
        if recompute:
            # recalculate attention score
            scores = torch.exp(torch.matmul(x, x.transpose(-2, -1))) \
                     / math.sqrt(x.size(-1))
            # mask missing points before softmax
            if type(mask) != type(None):
                mask_attn = attention_mask(mask.to(device), x.size(1))
            if aggregate:
                # Assume that the covariates dependencies are transferrable 
                p_attn = self.calculate_running_score(scores, mask_attn)
            else:
                if type(mask) != type(None):
                    scores = scores.masked_fill(mask_attn.to(device) == 0, 0)
                p_attn = F.softmax(scores, dim=-1)
                
            del scores, mask_attn

    #         if dropout is not None:
            p_attn = self.dropout(p_attn)
        
        else:
            # using stored scores
            scores = self.running_score.detach()
            p_attn = F.softmax(scores, dim=-1)
            p_attn = self.dropout(p_attn)
            
        if mask is not None:
#             print(p_attn.size(), x.size(), mask.size())
            output = torch.matmul(p_attn, x*mask.unsqueeze(-1).to(device))
            del mask
        else:
            output = torch.matmul(p_attn, x)
        
        if flatten:
#             return self.flatten(output), p_attn
#             print(scores.size())
            return output.sum(1), p_attn.detach()
        else:
            return output, p_attn.detach()
        
    def calculate_running_score(self, scores, mask_attn=None):
#         print(scores.size())
        if type(mask_attn) != type(None):
            scores = scores.masked_fill(mask_attn.to(device) == 0, 0)
            scores_nonzero_mean = (scores.sum(0)/(mask_attn.sum(0)).to(scores.device))
            del mask_attn
        else:
            scores_nonzero_mean = scores.mean(0)
        
        if self.running_score is not None > 0:
            new_scores = (1-self.alpha)*self.running_score + self.alpha*scores_nonzero_mean
        else:
            new_scores = scores_nonzero_mean
        del scores, scores_nonzero_mean
                
        p_attn = (F.softmax(new_scores.repeat(len(x),1,1), dim=-1))
        self.running_score = torch.nn.Parameter(new_scores, requires_grad=False)
        return p_attn
        
 
    
# class SelfAttention(nn.Module):
#     """
#     Compute 'Scaled Dot Product Attention
#     """
#     def __init__(self, ncov, dropout=0.1, alpha = 0.9):
#         super(SelfAttention, self).__init__()
#         self.flatten = nn.Flatten()
#         self.dropout = nn.Dropout(p=dropout)
#         self.alpha = alpha
#         self.running_score = torch.nn.Parameter(torch.zeros(ncov, ncov), requires_grad=False)
#     def forward(self, x, mask=None, flatten=True, aggregate=True, recompute=True):

#         # scores = torch.matmul(query, key.transpose(-2, -1)) \
#         #          / math.sqrt(query.size(-1))
        
#         # f(query, key)
#         # query: d \times m
#         # (d x m ) x (m x d)
#         # d \times d
#         device = x.device
#         if recompute:
#             # recalculate attention score
#             scores = torch.exp(torch.matmul(x, x.transpose(-2, -1))) \
#                      / math.sqrt(x.size(-1))
# #             print(scores.size())

#             if aggregate:
#                 # Assume that the covariates dependencies are transferrable

#                 # mask missing points before softmax
#                 if type(mask) != type(None):
#                     mask_attn = attention_mask(mask.to(device), x.size(1))
                    
#                 p_attn = self.calculate_running_score(scores, mask_attn)
# #                 print(scores.size())
#             else:
#                 if type(mask) != type(None):
#                     scores = scores.masked_fill(mask_attn.to(device) == 0, 0)
#                 p_attn = F.softmax(scores, dim=-1)
                

# #             del scores, mask_attn

#     #         if dropout is not None:
#             p_attn = self.dropout(p_attn)
        
#         else:
#             # using stored scores
#             scores = self.running_score.detach()
#             p_attn = F.softmax(scores, dim=-1)
#             p_attn = self.dropout(p_attn)
            
#         if mask is not None:
#             output = torch.matmul(p_attn, x*mask.unsqueeze(-1).to(device))
#             del mask
#         else:
#             output = torch.matmul(p_attn, x)
        
#         if flatten:
# #             return self.flatten(output), p_attn
# #             print(scores.size())
#             return output.sum(1), p_attn.detach()
#         else:
#             return output, p_attn.detach()
        
#     def calculate_running_score(self, scores, mask_attn=None):
# #         print(scores.size())
#         if type(mask_attn) != type(None):
#             scores = scores.masked_fill(mask_attn.to(device) == 0, 0)
#             scores_nonzero_mean = (scores.sum(0)/(mask_attn.sum(0)).to(scores.device))
#             del mask_attn
#         else:
#             scores_nonzero_mean = scores.mean(0)
        
#         if self.running_score.sum() > 0:
#             new_scores = (1-self.alpha)*self.running_score + self.alpha*scores_nonzero_mean
#         else:
#             new_scores = scores_nonzero_mean
#         del scores, scores_nonzero_mean
                
#         p_attn = (F.softmax(new_scores.repeat(len(x),1,1), dim=-1))
#         self.running_score = torch.nn.Parameter(new_scores, requires_grad=False)
#         return p_attn

    
    
    
# class SelfAttention(nn.Module):
#     """
#     Compute 'Scaled Dot Product Attention
#     """
#     def __init__(self, ncov, dropout=0.1, alpha = 0.9):
#         super(SelfAttention, self).__init__()
#         self.flatten = nn.Flatten()
#         self.dropout = nn.Dropout(p=dropout)
#         self.alpha = alpha
#         self.running_score = torch.nn.Parameter(torch.zeros(ncov, ncov), requires_grad=False)
#     def forward(self, x, mask=None, flatten=True, aggregate=True, recompute=True):

#         # scores = torch.matmul(query, key.transpose(-2, -1)) \
#         #          / math.sqrt(query.size(-1))
        
#         # f(query, key)
#         # query: d \times m
#         # (d x m ) x (m x d)
#         # d \times d
#         device = x.device
#         if recompute:
#             # recalculate attention score
#             scores = torch.exp(torch.matmul(x, x.transpose(-2, -1))) \
#                      / math.sqrt(x.size(-1))
# #             print(scores.size())

#             if aggregate:
#                 # Assume that the covariates dependencies are transferrable

#                 # mask missing points before softmax
#                 if type(mask) != type(None):
#                     mask_attn = attention_mask(mask.to(device), x.size(1))
                    
#                 p_attn = self.calculate_running_score(scores, mask_attn)
# #                 print(scores.size())
#             else:
#                 if type(mask) != type(None):
#                     scores = scores.masked_fill(mask_attn.to(device) == 0, 0)
#                 p_attn = F.softmax(scores, dim=-1)
                

# #             del scores, mask_attn

#     #         if dropout is not None:
#             p_attn = self.dropout(p_attn)
        
#         else:
#             # using stored scores
#             scores = self.running_score.detach()
#             p_attn = F.softmax(scores, dim=-1)
#             p_attn = self.dropout(p_attn)
            
#         if mask is not None:
#             output = torch.matmul(p_attn, x*mask.unsqueeze(-1).to(device))
#             del mask
#         else:
#             output = torch.matmul(p_attn, x)
        
#         if flatten:
# #             return self.flatten(output), p_attn
# #             print(scores.size())
#             return output.sum(1), p_attn.detach()
#         else:
#             return output, p_attn.detach()
        
#     def calculate_running_score(self, scores, mask_attn=None):
# #         print(scores.size())
#         if type(mask_attn) != type(None):
#             scores = scores.masked_fill(mask_attn.to(device) == 0, 0)
#             scores_nonzero_mean = (scores.sum(0)/(mask_attn.sum(0)).to(scores.device))
#             del mask_attn
#         else:
#             scores_nonzero_mean = scores.mean(0)
        
#         if self.running_score.sum() > 0:
#             new_scores = (1-self.alpha)*self.running_score + self.alpha*scores_nonzero_mean
#         else:
#             new_scores = scores_nonzero_mean
#         del scores, scores_nonzero_mean
                
#         p_attn = (F.softmax(new_scores.repeat(len(x),1,1), dim=-1))
#         self.running_score = torch.nn.Parameter(new_scores, requires_grad=False)
#         return p_attn
        
 
        
 

        
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
    