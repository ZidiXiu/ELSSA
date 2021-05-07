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
    
def batch_t_categorize(batch_t, batch_e, tt):
    nbin = len(tt)
    nbatch = batch_t.shape[0]
    all_cat = [t_categorize(batch_t[obs], batch_e[obs], tt) for obs in np.arange(nbatch)]  
    return torch.vstack(all_cat)


# get summary statistics of the predicted distribution
def calculate_quantiles(post_prob, tt, percentiles):
    post_prob_sums = torch.cumsum(post_prob,axis=1)
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