import math
import os

import numpy as np
import pandas

import scipy
import seaborn as sns

from time import time,sleep

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
# from torchmeta.modules import MetaModule
from collections import OrderedDict

from pathlib import Path

def landmarks_init(ncov, m, cts_var=None, cts_idx = None, device='cpu', x_landmarks='None', dataset=None):
    '''
    continuous_idx: 
    
    '''
    if type(cts_var) == type(None):
        cts_idx = np.arange(ncov)
        cts_var = cts_idx
    
    if type(x_landmarks) == type(None):
        # creat landmarks as traditional dictionary
        x_landmarks = {}
        '''based on each covariate percentile'''
    #     [x_landmarks.update({var: torch.tensor(np.percentile(dataset['x'][:,c_idx], np.linspace(0,100,m))).to(device).contiguous()}) for c_idx, var in zip(cts_idx, cts_var)]
    #     print(x_landmarks)
        '''based on the each covariate range'''
        [x_landmarks.update({var: torch.tensor(np.linspace(np.min(dataset['x'][:,c_idx]),np.max(dataset['x'][:,c_idx]),m, endpoint=True)).to(device).contiguous()}) for c_idx, var in zip(cts_idx, cts_var)]

    # save the initialized landmarks as a torch parameter dictionary
    x_emb_landmarks = nn.ParameterDict({})
    
    for var in cts_var:
        m = len(x_landmarks[var])
        x_emb_landmark = torch.eye(m)
        new_dict = nn.ParameterDict({var:torch.nn.Parameter(x_emb_landmark)})
        x_emb_landmarks.update(new_dict)
        
    return x_landmarks, x_emb_landmarks


def linear_interpolation_var(x, x_landmark, x_emb_landmark, m):
    # written for torch
    # returns index
    # a[i-1] < v <= a[i]
    indx = torch.searchsorted(x_landmark, x)
    # combine the first two indices, and the last two indices
    # to include the unobserved minimum and maximum value
    # combines [0,1], and [m-1, m]
    indx = torch.where(indx==0, 1, indx)
    indx = torch.where(indx==m, m-1, indx)

    coef = (x_emb_landmark[indx] - x_emb_landmark[indx-1])/(x_landmark[indx] - x_landmark[indx-1]).view(len(x),1)
    out = x_emb_landmark[indx-1] + coef*(x-x_landmark[indx-1]).view(len(x),1)    # linear interpolation
    return out


def cts_interpolation(x, x_landmarks, x_emb_landmarks, cts_var=None, cts_idx=None):
    
#     m = len(x_landmarks[0])
    if type(cts_var) == type(None):
        cts_var = x_landmarks.keys()
    # find values variable by variable
    res = []
    for c_idx, var in zip(cts_idx, cts_var):
        m = len(x_landmarks[var])
        cur_emb = linear_interpolation_var(x[:,c_idx], torch.tensor(x_landmarks[var]).to(x.device), x_emb_landmarks[var], m)
        # reshape to dimension [batch_size, 1, m]
        res.append(cur_emb.view(-1,1,m))
        
    return torch.cat(res, 1)
#     return torch.cat(res, 1)



def level_init(m, cat_var=None, cat_idx=None, x_levels=None, dataset=None):
#     if type(categorical_variables)==type(None):
#         categorical_variables = np.arange(len(x_levels))
    if type(x_levels) == type(None):
        x_levels = {} 
        [x_levels.update({var:len(np.unique(dataset['x'][:,c_idx]))}) for c_idx, var in zip(cat_idx, cat_var)]
    #     [x_levels.update({var:int(np.max(dataset['x'][:,c_idx]))}) for c_idx, var in zip(cat_idx, cat_var)]

    # save the initialized landmarks as a dictionary
    x_emb_levels = nn.ParameterDict({})
    for var in cat_var:
        level = x_levels[var]
        # randomly pick one position as 1
        idx = torch.randint(m,(level,1)).squeeze()
        x_emb_level = torch.zeros(level, m)
        x_emb_level[np.arange(level).tolist(),idx] = 1
        
#         x_emb_level = torch.rand(level, m)
#         x_emb_level = F.softmax(torch.rand(level, m),dim=-1)
    
        new_dict = nn.ParameterDict({var:torch.nn.Parameter(x_emb_level)})
        x_emb_levels.update(new_dict)
    return x_levels, x_emb_levels


def cat_interpolation(x, x_emb_levels, cat_var=None, cat_idx = None, m=None):
    if type(cat_var) == type(None):
        cat_var = x_emb_levels.keys()
    # find values variable by variable
    res = []
    for c_idx, var in zip(cat_idx, cat_var):
#         print(c_idx, var)
        cur_emb = x_emb_levels[var][x[:,c_idx].long()]
        # reshape to dimension [batch_size, 1, m]
        res.append(cur_emb.view(-1,1,m))
        
    return torch.cat(res, 1)
    

    
def cov_embedding(x, m, cts_var=None, cts_idx = None, x_landmarks=None, x_emb_landmarks=None,
                  cat_var=[], cat_idx = [], x_levels=None, x_emb_levels=None):
    '''
    Do the transformation within each minibatch

    m: length of the embedding vector for each covariates

    cts_var: list of the names of continuous variables
    cat_var: list of the names of categorical variables

    x_landmarks: dictionary with continuous names as keys, and the percentile landmarks as items
    x_emb_landmarks: dictionary with continuous names as keys, and the embedding landmarks as items

    x_levels: dictionary with categorical names as keys, and the number of levels per covariate as items
    x_emb_levels: dictionary with categorical names as keys, and the embedding vectors as items
    '''
    x_emb = []
    var_list = []
    if type(cts_var) != type(None):
        x_emb.append(cts_interpolation(x, x_landmarks, x_emb_landmarks, cts_var, cts_idx))
        var_list.extend(list(x_emb_landmarks.keys()))
        
    if len(cat_var) >0:
        x_emb.append(cat_interpolation(x, x_emb_levels, cat_var, cat_idx, m))
        var_list.extend(list(x_emb_levels.keys()))
    
    x_emb = torch.cat(x_emb, 1)
#     cur_var_list = torch.cat(var_list)
#     print(var_list)
    return x_emb, var_list
        
    
class Embedding(nn.Module):
    '''
    create embedding paramters
    '''
    def __init__(self, ncov, cts_var, cts_idx, cat_var=[], cat_idx=[], x_landmarks = None, x_levels=None, dataset=None, dropout=0.0, m=10, device='cuda'):
        super(Embedding, self).__init__()
        self.m = m
        self.cts_var, self.cts_idx, self.cat_var, self.cat_idx =  cts_var, cts_idx, cat_var, cat_idx
        
        # Initiate landmarks with trainable paramters
        self.x_landmarks, self.x_emb_landmarks = landmarks_init(ncov, m, cts_var, cts_idx, device, x_landmarks, dataset)
        if len(cat_idx) > 0:
            self.x_levels, self.x_emb_levels = level_init(m, cat_var, cat_idx, x_levels, dataset)
        else:
            self.x_levels, self.x_emb_levels = None, None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x, mask=None):

        x_emb, var_list = cov_embedding(x.float(), self.m,\
                                        self.cts_var, self.cts_idx, self.x_landmarks, self.x_emb_landmarks,\
                                        self.cat_var, self.cat_idx, self.x_levels, self.x_emb_levels)
#         print(var_list)
        return self.dropout(x_emb).float(), var_list

