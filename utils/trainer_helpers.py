import math
import os

import numpy as np
import pandas

import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F

from time import time,sleep

# discretized time into bins
# helper functions
# transform t into categories
# one-hot-encoding
# only consider the nearest category to the left
##### for re-formulate time #####
def t_tt_idx(t_,tt):
    t_diff = t_-tt
    t_diff_sign = (t_diff>=0)
    if len(t_diff[t_diff_sign])>0:
        idx = torch.argmin(t_diff[t_diff_sign])
    else:
        idx = 0
    return idx


def t_label(t_,tt):
    nbin = len(tt)
    t_diff = t_-tt
    t_diff_sign = (t_diff>=0)
    if len(t_diff[t_diff_sign])>0:
        idx = torch.argmin(t_diff[t_diff_sign])
        t_cat = torch.zeros(nbin).to(t_.device)
        t_cat[idx] = 1.0
    else:
        t_cat = torch.zeros(nbin).to(t_.device)
        t_cat[0] = 1.0
    return t_cat

def t_label_censor(t_, tt):
    t_idx = t_tt_idx(t_,tt)
    t_prob = torch.cat([torch.zeros(t_idx), torch.ones(len(tt)-t_idx)])
    return t_prob.to(t_.device)


def t_categorize(t_, e_, tt):
    if e_ == 1:
        t_cat = t_label(t_,tt)
    else:
        t_cat = t_label_censor(t_, tt)
    return t_cat

# # previous version, rounded to the nearist bin
def batch_t_categorize(batch_t, batch_e, tt):
    nbin = len(tt)
    nbatch = batch_t.shape[0]
    all_cat = [t_categorize(batch_t[obs], batch_e[obs], tt) for obs in np.arange(nbatch)]  
    return torch.vstack(all_cat)

def time_embedding(batch_t, batch_e, tt):

    # written for torch
    # returns index
    # a[i-1] < v <= a[i]
    nbin = len(tt)
    time_emb_landmark = torch.eye(nbin).to(batch_t.device)
    indx = torch.searchsorted(tt, batch_t)
    # combine the first two indices, and the last two indices
    # to include the unobserved minimum and maximum value
    # combines [0,1], and [m-1, m]
    indx = torch.where(indx==0, 1, indx)
    indx = torch.where(indx==nbin, nbin-1, indx)
    # for event
    coef = (time_emb_landmark[indx] - time_emb_landmark[indx-1])/(tt[indx].to(batch_t.device) - tt[indx-1].to(batch_t.device)).view(len(indx),1)
#         print(coef.size())
    out = time_emb_landmark[indx-1] + coef*(batch_t-tt[indx-1].to(batch_t.device)).view(len(indx),1)    # linear interpolation

    # for censoring
    censor_mask = torch.vstack([(torch.cat((out[i,:idx+1].squeeze(), torch.ones(nbin-idx-1).to(batch_t.device))).view(1, nbin)) for i, idx in enumerate(indx)]).float()
    out_censor = torch.matmul(censor_mask, time_emb_landmark.T)

    del censor_mask, coef

    return (out*batch_e.view(len(out),1) + out_censor*(1-batch_e).view(len(out),1)).float()

        
# get summary statistics of the predicted distribution
def calculate_quantiles(post_prob, tt, percentiles):
    post_prob_sum = torch.cumsum(post_prob,axis=1)
    try:
        tt_p = [tt[torch.argmin(torch.abs(post_prob_sum-p))] for p in percentiles]
    except TypeError:
        tt_p = tt[torch.argmin(torch.abs(post_prob_sum-percentiles))]
        tt_p = [tt_p]
        
    return tt_p

def get_median(p_raw, tt, percentile):
    post_prob_sums = torch.cumsum(p_raw,axis=1)
    tt_idx = torch.argmin(torch.abs(post_prob_sums-percentile),axis=1)
    t_med_hat = tt[tt_idx]
    return t_med_hat


def wt_avg(p_raw, tt):
    # calculate the averaged sum of these bins as a summary
    t_wa_hat = (p_raw*tt).sum(1)
    return t_wa_hat


# def attention_mask(mask_new, ncov, p=0.1):
#     mask_attn = (torch.ones(len(mask_new), ncov, ncov).to('cpu')-torch.diag(torch.ones(ncov)).unsqueeze(0).repeat(len(mask_new),1,1).to('cpu')+torch.vstack([torch.diag(mask_new[idx]).unsqueeze(0) for idx in range(len(mask_new))]))*(torch.rand(len(mask_new), ncov, ncov) > p).to('cpu')
#     return mask_attn

def attention_mask(mask_new, ncov):
    mask_attn = torch.vstack([((mask_new[idx].unsqueeze(1).repeat(1, ncov))*(mask_new[idx].unsqueeze(0).repeat(ncov,1))).unsqueeze(0) for idx in range(len(mask_new))])
    return mask_attn
#     if p > 0:
#         return mask_attn*(torch.rand(len(mask_new), ncov, ncov) > p).to('cpu')
#     else:
        

# def attention_mask(mask_new, ncov, p=0.1, aggregate = True):
#     mask_attn = torch.vstack([((mask_new[idx].unsqueeze(1).repeat(1, ncov))*(mask_new[idx].unsqueeze(0).repeat(ncov,1))).unsqueeze(0) for idx in range(len(mask_new))])
#     if aggregate = True:
        
#     if p > 0:
#         return mask_attn*(torch.rand(len(mask_new), ncov, ncov) > p).to('cpu')
#     else:
#         return mask_attn
