
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
# def NLL_reg(p_raw, y, e, tt, collapsed=True):
#     # using likelihood to regularize the performance
#     y_cat = batch_t_categorize(y, e, tt)
#     #         keep_idx = torch.where(y <= t_max)[0]
    
#     if collapsed:
#         y_loglikeli = -((p_raw*torch.tensor(y_cat)).sum(axis=1)+1e-6).log().sum()
#     else:
#         y_loglikeli = -((p_raw*torch.tensor(y_cat)).sum(axis=1)+1e-6).log()
#     #  -((p_raw*torch.tensor(y_cat)).sum(axis=1)+1e-4).log().mean()
    
#     return y_loglikeli

def NLL_reg(p_raw, y, e, tt, collapsed=True):
    # using likelihood to regularize the performance
    y_cat = time_embedding(y, e, tt)
    #         keep_idx = torch.where(y <= t_max)[0]
    
    if collapsed:
        y_loglikeli = -((p_raw*torch.tensor(y_cat)).sum(axis=1)+1e-6).log().sum()
    else:
        y_loglikeli = -((p_raw*torch.tensor(y_cat)).sum(axis=1)+1e-6).log()
    #  -((p_raw*torch.tensor(y_cat)).sum(axis=1)+1e-4).log().mean()
    
    return y_loglikeli


def NLL_reg_emb(p_raw, y, e, tt, collapsed=True):
    # using likelihood to regularize the performance
    y_cat = time_embedding(y, e, tt)
    #         keep_idx = torch.where(y <= t_max)[0]
    
    if collapsed:
        y_loglikeli = -((p_raw*torch.tensor(y_cat)).sum(axis=1)+1e-6).log().sum()
    else:
        y_loglikeli = -((p_raw*torch.tensor(y_cat)).sum(axis=1)+1e-6).log()
    #  -((p_raw*torch.tensor(y_cat)).sum(axis=1)+1e-4).log().mean()
    
    return y_loglikeli


def get_CI_raw(event, true_t, pred_t, torch_object=False):
    if torch_object:
        return concordance_index_censored((event.squeeze().cpu().detach().numpy()).astype(bool), true_t.squeeze().cpu().detach().numpy(), -pred_t.squeeze().cpu().detach().numpy())
    
    else:
        return concordance_index_censored(event.astype(bool), true_t, -pred_t)

# calculate point estimation loss
def point_loss(t_hat, y, e, loss_type='MSE', return_sum = True):
    # point estimation loss based on the predicted raw probabilities
    hinge_loss,_ = torch.min(torch.cat([t_hat.unsqueeze(1)-y.unsqueeze(1), torch.zeros_like(t_hat.unsqueeze(1)).to(y.device)], dim=1), 1, keepdim=True)
    
    
    if loss_type == 'MSE':
        eloss = (e*torch.pow((y-t_hat.squeeze()),2)).sum()
        closs = ((1-e)*torch.pow(hinge_loss.squeeze(),2)).sum()
    elif loss_type == 'MAE':
        eloss = (e*torch.abs(y-t_hat.squeeze())).sum()
        closs = ((1-e)*torch.abs(hinge_loss.squeeze())).sum()
    elif loss_type == 'RAE':
        eloss = (e*torch.abs(y-t_hat.squeeze())/y.squeeze()).sum()
        closs = ((1-e)*torch.abs(hinge_loss.squeeze())/y.squeeze()).sum()
        
    if return_sum:
        return eloss, closs
    else:
        len_e = len(e[e==1])
        return eloss/len_e, closs/(len(e)-len_e)

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

# for the coverage plot
def calculate_quantiles(post_prob, tt, percentiles):
    post_prob_sum = np.cumsum(post_prob)
    try:
        tt_p = [tt[np.argmin(np.abs(post_prob_sum-p))] for p in percentiles]
    except TypeError:
        tt_p = tt[np.argmin(np.abs(post_prob_sum-percentiles))]
        tt_p = [tt_p]
        
    return(np.array(tt_p))


def calculate_coverage(pred_prob, tt, t_,quantiles, aft_model=False):
    if not aft_model:
        ci_list = [calculate_quantiles(post_prob,tt,quantiles) for post_prob in pred_prob]
        ci_list = np.array(ci_list)
    if aft_model:
        ci0 = aft.predict_percentile(pred_prob, p=quantiles[0]).values 
        ci1 = aft.predict_percentile(pred_prob, p=quantiles[1]).values
        ci_list = np.concatenate([ci1, ci0], axis=1)
        
    coverage_list = []
    for i in np.arange(len(t_)):
        if (t_[i]>=ci_list[i,0]) and (t_[i]<=ci_list[i,1]):
            coverage_list.append(1)            
        else:
            coverage_list.append(0)
    return(coverage_list)

def calculate_coverage_censor(pred_prob, tt, t_,quantiles, aft_model=False):
    if not aft_model:
        ci_list = [calculate_quantiles(post_prob,tt,quantiles) for post_prob in pred_prob]
        ci_list = np.array(ci_list)
    if aft_model:
        ci0 = aft.predict_percentile(pred_prob, p=quantiles).values 
        ci_list = np.array(ci0).reshape(len(t_))
    coverage_list = []
#     print(ci_list)
    for i in np.arange(len(t_)):
        if (t_[i]<=ci_list[i]):
            coverage_list.append(1)            
        else:
            coverage_list.append(0)
    return(coverage_list)
