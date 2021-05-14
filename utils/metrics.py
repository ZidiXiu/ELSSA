
import math
import os

import numpy as np
import pandas

import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F

from time import time,sleep

from sksurv.metrics import concordance_index_censored
from utils.trainer_helpers import *


# calculate negative likelihood 
def NLL_reg(p_raw, y, e, tt, collapsed=True):
    # using likelihood to regularize the performance
    y_cat = batch_t_categorize(y, e, tt)
    #         keep_idx = torch.where(y <= t_max)[0]
    
    if collapsed:
        y_loglikeli = -((p_raw*torch.tensor(y_cat)).sum(axis=1)+1e-6).log().sum()
    else:
        y_loglikeli = -((p_raw*torch.tensor(y_cat)).sum(axis=1)+1e-6).log()
    #  -((p_raw*torch.tensor(y_cat)).sum(axis=1)+1e-4).log().mean()
    
    return y_loglikeli


def get_CI_raw(event, true_t, pred_t):
    return concordance_index_censored((event.squeeze().cpu().detach().numpy()).astype(bool), true_t.squeeze().cpu().detach().numpy(), -pred_t.squeeze().cpu().detach().numpy())

# calculate point estimation loss
def point_loss(t_hat, y, e, loss_type='MSE'):
    # point estimation loss based on the predicted raw probabilities
    hinge_loss,_ = torch.min(torch.cat([t_hat.unsqueeze(1)-y.unsqueeze(1), torch.zeros_like(t_hat.unsqueeze(1)).to(y.device)], dim=1), 1, keepdim=True)
    
    
    if loss_type == 'MSE':
        eloss = (e*torch.pow((y-t_hat.squeeze()),2)).sum()
        closs = ((1-e)*torch.pow(hinge_loss.squeeze(),2)).sum()
    elif loss_type == 'MAE':
        eloss = (e*torch.abs(y-t_hat.squeeze())).sum()
        closs = ((1-e)*torch.abs(hinge_loss.squeeze())).sum()
    elif loss_type == 'RAE':
        eloss = (e*torch.abs(y-t_hat.squeeze())/t_hat.squeeze()).sum()
        closs = ((1-e)*torch.abs(hinge_loss.squeeze())/t_hat.squeeze()).sum()
        
    return eloss, closs

'''time dependent Concordance Index'''

def I_Ctd_DLN(t, e, test_pred_prob, tt, i,j):
#     x_i = x[i]
#     x_j = x[j]
    t_true_i = t[i]
    t_i_idx = torch.where(batch_t_categorize(t[i].reshape([1,1]), e[i].reshape([1,1]), tt)[-1]==1)[0]
    sum_idx = torch.cat([torch.ones(t_i_idx), torch.zeros(len(tt)-t_i_idx)])
#     print(test_pred_prob[i], sum_idx)
    F_i = torch.dot(test_pred_prob[i].squeeze(), sum_idx)
    F_j = torch.dot(test_pred_prob[j].squeeze(), sum_idx)
    return(1*(F_i > F_j).cpu().detach().item())
    # return (log_S_i, log_S_j)

def pair_Ctd_DLN(t, e, test_pred_prob, tt):
    j_pool = []
    while len(j_pool)==0:
        subj_i = np.random.choice(torch.where(e==1)[0],1)
        j_pool = torch.where(t>t[subj_i])[0]
        
    subj_j = np.random.choice(torch.where(t>t[subj_i])[0],1)
        
    return(I_Ctd_DLN(t, e, test_pred_prob, tt, subj_i,subj_j))