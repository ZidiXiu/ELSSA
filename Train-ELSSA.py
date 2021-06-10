#!/usr/bin/env python
# coding: utf-8

# Attention Layer with Contrastive Learning

# In[1]:


from __future__ import absolute_import, division, print_function

import math
import os

import numpy as np
import pandas

import scipy
import seaborn as sns

from time import time,sleep

import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
from torch import optim

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler

from pathlib import Path

import warnings
warnings.filterwarnings('ignore')

mypal = sns.color_palette('Set2')
emp_color = mypal[0]
pred_color = mypal[1]
print ("PyTorch version: " + torch.__version__)


# In[2]:


from data.utils import SimpleDataset

from utils.metrics import NLL_reg, point_loss, get_CI_raw, pair_Ctd_DLN
from utils.trainer_helpers import batch_t_categorize, wt_avg, attention_mask


from data.dialysisData_mask import generate_data
from data.utils import SimpleDataset, SimpleDataset_masked

import argparse

parser = argparse.ArgumentParser(description='PyTorch ELSSA Training')
parser.add_argument('--dataset', default='SEER', help='dataset setting')
parser.add_argument('-s',  default=32, type=int, help='embedding dimensions')
parser.add_argument('-nbin', default=100, type=int, help='discrete bins for time-to-event')

# parser.add_argument('--loss_type', default="MSE", type=str, help='point estimation loss type')
parser.add_argument('--loss_type',
                    default='MSE',
                    const='MSE',
                    nargs='?',
                    choices=['MSE', 'MAE', 'RAE'],
                    help='point estimation loss type (default: MSE)')

parser.add_argument('--percentile', default="True", type=str, help='time line discretize method')
parser.add_argument('--event_based', default="False", type=str, help='based on event timeline only')
parser.add_argument('--num_try', default="0", type=str, help='number of try for the current network')

parser.add_argument('-enc_dim', nargs='+', type=int, default=[32,32],
                    help='encoder structure')
parser.add_argument('-dec_dim', nargs='+', type=int, default=[32,32],
                    help='decoder structure')

parser.add_argument('--emb_lr', default=1e-4, type=float,
                    metavar='EMBLR', help='learning rate for embedding network')
parser.add_argument('--clf_lr', default=1e-4, type=float,
                    metavar='CLFLR', help='learning rate for contrastive network')
parser.add_argument('--dec_lr', default=1e-4, type=float,
                    metavar='DECLR', help='learning rate for decoding network')

parser.add_argument('--ct_wt', default=1, type=float, help='weight for contrastive loss')
parser.add_argument('--pt_wt', default=0, type=float, help='weight for point estimation loss')

parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--continue_training', default='False', type=str,
                    help='continue training on the last best epoch')
parser.add_argument('-batch_size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 100)')

parser.add_argument('--result_path_root', '--wording-dir',default='', type=str, metavar='PATH',
                    help='path to the result files')
parser.add_argument('--file_path',default='', type=str, metavar='FILEPATH',
                    help='path to the data files')
parser.add_argument('-evaluate', default="True", type=str, help='evaluate model on validation set')
parser.add_argument('-training', default="True", type=str, help='train the model on training set')

parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--num_workers', default=5, type=int,
                    help='number of workers for DataLoader.')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')


# # to run argparse without command line
# import sys
# sys.argv = ['']

global args
args = parser.parse_args()


if args.dataset == 'Dialysis':
    args.file_path = file_path='/storage/zx35-dgim1/User_Projects/Pro00105834_Goldstein/AnalyticData'

    args.train_file = 'DCITrain03252021.csv'
    args.valid_file = 'DCIValidation03252021.csv'
# else:
#     args.train_file = 'DCITrain03252021.csv'
#     args.valid_file = 'DCIValidation03252021.csv'


args.model_name = 'ELSSA'


event_based = 'event' if args.event_based == 'True' else 'all'
percentile = 'percentile' if args.percentile == 'True' else 'linespace'

args.store_name = '_'.join([args.model_name, args.dataset, args.num_try,\
                            event_based,percentile, 'nbin', str(args.nbin),\
                            'emb', str(args.s),\
                            'enc', '_'.join([str(dim) for dim in args.enc_dim]),\
                            'dec', '_'.join([str(dim) for dim in args.dec_dim]),\
                           'ctWt', str(args.ct_wt), 'ptWt', str(args.pt_wt)])

print(args.store_name)

train, valid, variable_info = generate_data(args.file_path, args.train_file, args.valid_file, m=args.s)
train_size = train['x'].shape[0]
val_size = valid['x'].shape[0]


cov_list, cts_var, cts_idx, cat_var, cat_idx = variable_info['cov_list'], variable_info['cts_var'], variable_info['cts_idx'], variable_info['cat_var'], variable_info['cat_idx']

x_landmarks, x_levels = variable_info['x_landmarks'], variable_info['x_levels']



if args.event_based == 'True':
    train_time = train['t'][train['e']==1]
else:
    train_time = train['t']

if args.percentile == 'True':
    '''based on each covariate percentile'''
    t_landmarks = torch.tensor(np.percentile(train_time, np.linspace(0,100,args.s))).contiguous()
    tt = np.percentile(train_time,np.linspace(0.,100.,args.nbin, endpoint=True))

else:
    '''based on the each covariate range'''
    t_landmarks = torch.tensor(np.linspace(np.min(train_time),np.max(train_time),args.s, endpoint=True)).contiguous()
    tt = np.linspace(np.min(train_time),np.max(train_time),args.nbin, endpoint=True)


# based on whether we have censoring after the largest observed t
loss_of_info = np.mean(train['t']>np.max(train['t'][train['e']==1]))

# need to convert t to different size of bins
if loss_of_info > 0.0001:
    args.nbin = args.nbin + 1
    # add the largest observed censoring time inside
    tt = np.append(tt,np.max(train['t']))
#     event_tt_prob = risk_t_bin_prob(train['t'], train['e'], tt)


# In[8]:

plot_path = args.result_path_root + '/plot'
Path(plot_path).mkdir(parents=True, exist_ok=True)

plt.figure()
sns.distplot(train['t'][train['e']==1])
plt.plot(tt, np.zeros(len(tt)),'o')
plt.savefig(plot_path+'/marginal_t.png')



# ### Embedding the covariates into $\mathbb{R}^m$
# - for each continuous variable, first find $m$ landmarkers, then interpolate values in between
# - for each categorical variable, find the matched embedding vector with $m$ dimensions
# 
# Define landmarkers for each continuous variable### Embedding the covariates into $\mathbb{R}^m$

# In[9]:




import torch
from torch import nn, optim
import numpy as np

# Type hinting
from typing import Union, List, Optional, Any, Tuple
from torch import FloatTensor, LongTensor

# load embedding and attention networks
from model.embedding import Embedding
from model.attention import Attention, SelfAttention
from model.SimpleNN import DecMLP_bin, SimpleMLP
from model.ContrastiveLearning import FDV_CL


 
ncov = train['x'].shape[1]
covList = np.arange(ncov)

trainData = SimpleDataset_masked(train['x'], train['t'], train['e'], train['missing_mask'])
pair1 = DataLoader(trainData, batch_size=args.batch_size,shuffle=True, num_workers = args.num_workers)


validData = SimpleDataset_masked(valid['x'], valid['t'], valid['e'], valid['missing_mask'])
v_pair1 = DataLoader(validData, batch_size=args.batch_size,shuffle=True, num_workers =  args.num_workers)
# v_pair2 = DataLoader(validData, batch_size=500,shuffle=True)

# testData = SimpleDataset_masked(test['x'], test['t'], test['e'], test['missing_mask'])
# t_pair1 = DataLoader(testData, batch_size=1000,shuffle=True)


zdim = args.s
epochs = args.epochs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_name = args.gpu
n_gpu = torch.cuda.device_count()
all_device = set(np.arange(n_gpu))
all_device.remove(args.gpu)
device_ids = [args.gpu] + list(all_device)
print(device_ids)

torch.cuda.set_device(device_name)

# bined time
tt = torch.tensor(tt).to(device)

result_path = args.result_path_root+'/'+args.model_name+'/'
plot_path = result_path + 'plot'
model_path = result_path +'saved_models/'

Path(result_path).mkdir(parents=True, exist_ok=True)
Path(model_path).mkdir(parents=True, exist_ok=True)
Path(plot_path).mkdir(parents=True, exist_ok=True)

pickle.dump( args, open( args.result_path_root+"/"+args.model_name+"/"+args.store_name+"_args.pkl", "wb" ) )


'''covariate embedding'''
embedding = Embedding(ncov, cts_var, cts_idx, cat_var, cat_idx, x_landmarks, x_levels, m=args.s)

'''attention network'''
# attention = LinearAtt(ncov=ncov, dropout=0.1)
attention = SelfAttention(dropout=0.1)

'''contrastive learning'''
# input is a batch of embedded x
clf = FDV_CL(m = args.s, ncov = ncov, t_landmarks = t_landmarks, h_dim=args.enc_dim)

'''decoding the embedded vectors'''
decoder = DecMLP_bin(input_size = zdim, output_size = args.nbin, h_dim=args.dec_dim)


emb_path = model_path+args.store_name+'_emb'+'.pt'
dec_path = model_path+args.store_name+'_dec'+'.pt'
att_path = model_path+args.store_name+'_att'+'.pt'
clf_path = model_path+args.store_name+'_clf'+'.pt'

if args.continue_training == 'True':
    print('Load last best model')
    embedding.load_state_dict(torch.load(emb_path))
    attention.load_state_dict(torch.load(att_path))
    clf.load_state_dict(torch.load(clf_path))
    decoder.load_state_dict(torch.load(dec_path))
    
    
# att_path = result_path+model_name+'_att.pt'

del train, valid, trainData, validData


embedding.to(device)
clf.to(device)
attention.to(device)
decoder.to(device)

if args.training == 'True':
    train_plot_path = plot_path+'/train'
    Path(train_plot_path).mkdir(parents=True, exist_ok=True)

    # define optimizer
    opt_emb = optim.Adam(embedding.parameters(), lr=args.emb_lr)
    opt_dec = optim.Adam(decoder.parameters(), lr=args.dec_lr)
    opt_clf = optim.Adam(clf.parameters(), lr=args.clf_lr)

    # define scheduler
    emb_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt_emb, mode='min', factor=0.1, patience=4, min_lr=0, verbose=True)
    dec_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt_dec, mode='min', factor=0.1, patience=4, min_lr=0, verbose=True)
    clf_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt_clf, mode='min', factor=0.1, patience=2, min_lr=0, verbose=True)



    attention = nn.DataParallel(attention, device_ids=device_ids)
    decoder = nn.DataParallel(decoder, device_ids=device_ids)
    clf = nn.DataParallel(clf, device_ids=device_ids)

    print(embedding, attention, clf, decoder)

    # In[ ]:


    best_valid_e_loss = np.inf
    best_valid_NLL_loss = np.inf
    best_valid_CI = 0
    best_epoch = 0

    train_loss_hist = []
    train_loss_NLL_hist = []
    train_loss_est_hist = []
    train_loss_ctr_hist = []

    valid_loss_hist = []
    valid_loss_NLL_hist = []
    valid_loss_est_hist = []
    valid_loss_ctr_hist = []

    retrained = False
    for epoch in range(1, epochs + 1):
#     for epoch in range(cur_epoch, epochs + 1):


        train_loss = 0
        train_ctr_loss = 0
        train_NLL_loss, train_est_loss = 0, 0

        valid_e_loss = 0
        valid_c_loss = 0
        valid_NLL_loss, valid_est_loss = 0, 0

        print('epoch'+str(epoch))
        improved_str = " "
        embedding.train()
        attention.train()
        decoder.train()
        clf.train()


        for i, (x, y, e, mask) in enumerate(pair1):
            # training encoder and decoder

            x= x.to(device).float()
            y = y.to(device).float()
            e = e.to(device)

            x_emb, var_list = embedding(x.float())
            del x

            # re-indexing mask
            # when certain observation is missed, the corresponding mask is 0
            reindex = torch.tensor([np.where(cov_list==var)[0][0] for var in var_list])
            mask_new = torch.index_select(mask, 1, reindex)
            del mask, reindex
            mask_attn = attention_mask(mask_new, ncov, p=0.1)
            del mask_new
            z, attn_score = attention(x_emb, mask = mask_attn)

            del mask_attn, x_emb

            loss_infoNCE = (clf(z, y, e)).mean()

            p_raw = decoder(clf.module.enc(z))

            loss_NLL = NLL_reg(p_raw, y, e, tt)

            # weighted average
            t_wa_hat = wt_avg(p_raw, tt)

            del p_raw

            eloss, closs = point_loss(t_wa_hat, y, e, args.loss_type)
            loss_est = (closs + eloss)/len(e)

            surv_loss = loss_NLL + args.pt_wt*loss_est
            del t_wa_hat

            # update parameters
            loss = surv_loss + args.ct_wt*loss_infoNCE
            loss.backward()

            torch.nn.utils.clip_grad_norm_(embedding.parameters(), 1e-1)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 1e-1)
            torch.nn.utils.clip_grad_norm_(clf.parameters(), 1e-1)

            train_loss += surv_loss.item()
            train_NLL_loss += loss_NLL.item()    
            train_est_loss += loss_est.item()
            train_ctr_loss += loss_infoNCE.item()

            opt_emb.step()
            opt_clf.step()
            opt_dec.step()
            
            del loss, surv_loss, loss_infoNCE, loss_est


    #        if i > len(pair1)/2:
    #            break

        print('    training finished for current epoch')
        embedding.eval()
        attention.eval()
        decoder.eval()
        clf.eval()

        valid_e_loss = 0
        valid_c_loss = 0
        valid_ct_loss = 0
        valid_pred_t, valid_e, valid_t = [],[],[]

        for i, (x, y, e, mask) in enumerate(v_pair1):
            if i%1000 == 0:
                print('    iteration:' + str(i))

            x= x.to(device).float()
            y = y.to(device).float()
            e = e.to(device)


            x_emb, var_list = embedding(x.float())

            del x
            # re-indexing mask
            reindex = torch.tensor([np.where(cov_list==var)[0][0] for var in var_list])
            mask_new = torch.index_select(mask, 1, reindex)
            del mask, reindex
            mask_attn = attention_mask(mask_new, ncov, p=0.0)

            del mask_new
            z, attn_score = attention(x_emb, mask = mask_attn)
            del mask_attn, x_emb

            loss_infoNCE = (clf(z, y, e)).mean()
            valid_ct_loss += loss_infoNCE.item()
            del  loss_infoNCE
            p_raw = decoder(clf.module.enc(z))

            del z
            loss_NLL = NLL_reg(p_raw, y, e, tt)

            # weighted average
            t_wa_hat = wt_avg(p_raw, tt)
            # save one subject for plotting
            subj = np.random.choice(len(e))
            subj_res = t_wa_hat[subj].squeeze().detach().cpu().item(), p_raw[subj].detach().cpu().numpy(), y[subj].detach().cpu().item(), e[subj].detach().cpu().item()

            del p_raw

            eloss, closs = point_loss(t_wa_hat, y, e, args.loss_type)
            loss_est = (closs + eloss)/len(e)

    #         surv_loss = loss_NLL + args.pt_wt*loss_est

            valid_e_loss += eloss.item()
            valid_c_loss += closs.item()


            valid_NLL_loss += loss_NLL.item()
            valid_est_loss += loss_est.item()

            valid_pred_t.append(t_wa_hat.squeeze().detach().cpu().numpy())
            valid_t.append(y.squeeze().detach().cpu().numpy())
            valid_e.append(e.squeeze().detach().cpu().numpy())

            del t_wa_hat, y, e, loss_NLL, loss_est, eloss, closs

        # try scheduler
        emb_scheduler.step(valid_NLL_loss)
        clf_scheduler.step(valid_ct_loss)
        dec_scheduler.step(valid_est_loss)        

        # concatenate all the validation results
        valid_e = np.concatenate(valid_e)
        valid_t = np.concatenate(valid_t)
        valid_pred_t = np.concatenate(valid_pred_t)

        valid_CI = get_CI_raw(valid_e, valid_t, valid_pred_t, torch_object =False)[0]
        train_loss_hist.append(train_loss/train_size)
        train_loss_NLL_hist.append(train_NLL_loss/train_size)
        train_loss_est_hist.append(train_est_loss/train_size)
        train_loss_ctr_hist.append(train_ctr_loss/train_size)

        valid_loss_hist.append(valid_e_loss/val_size)
        valid_loss_NLL_hist.append(valid_NLL_loss/val_size)
        valid_loss_est_hist.append(valid_est_loss/val_size)
        valid_loss_ctr_hist.append(valid_ct_loss/val_size)

        save_model = 0
        if (best_valid_e_loss > valid_e_loss):
            save_model += 1
            best_valid_e_loss = valid_e_loss

        if (best_valid_NLL_loss > valid_NLL_loss):
            best_valid_NLL_loss = valid_NLL_loss
            save_model += 2
        if (best_valid_CI < valid_CI):
            save_model += 1
            best_valid_CI = valid_CI



        if save_model>1 or epoch % args.print_freq==0:

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)

            r_idx = np.random.choice(np.where(valid_e==1)[0],500)
            sns.scatterplot(valid_pred_t[r_idx], valid_t[r_idx])
            plt.plot(valid_t[r_idx], valid_t[r_idx])
            plt.xlabel('Predicted')
            plt.ylabel('True Event')

            plt.subplot(1, 2, 2)

            t_hat, p_raw, true_t, event_type = subj_res
            plt.axvline(true_t,linestyle='--',color=emp_color,label='observed t_k')
            plt.axvline(t_hat,linestyle='--',color=pred_color,label='predicted t_k')

            sns.scatterplot(tt.cpu().detach().numpy(), p_raw, label='Estimated', color=pred_color)

            plt.xticks(np.linspace(tt[0].item(),tt[-1].item(),10, endpoint=True))  # Set label locations.

            plt.title(r'$\delta$='+str(event_type))
            plt.savefig(train_plot_path+'/'+args.store_name+'_epoch'+str(epoch)+'.png')

        if save_model > 1:
            best_epoch = epoch

            torch.save(embedding.state_dict(), emb_path)
            torch.save(clf.module.state_dict(), clf_path)
            torch.save(decoder.module.state_dict(), dec_path)
            torch.save(attention.module.state_dict(), att_path)

            improved_str = "*"
        print('====> Train NLL: {:.3f} \t Valid NLL: {:.3f} CI: {:.3f} event loss: {:.3f} \t censoring loss : {:.3f} \t Improved: {}'.format(train_NLL_loss/train_size, valid_NLL_loss/val_size, valid_CI, valid_e_loss/val_size,valid_c_loss/val_size,improved_str))
        del valid_t, valid_e, valid_pred_t


    #     if ctr_learning and (epoch - best_epoch >=20):
    #         ctr_learning = False
    #         print('Contrastive learning stopped')

    #     if not ctr_learning and (epoch - best_epoch >=50):
    #         print('Model stopped due to early stopping')
    #         break

        if epoch - best_epoch >=10 and not retrained:
            print('Model stopped due to early stopping')
#             break
            print('Reload the current best model')
            best_epoch = epoch
            '''
            LOAD THE BEST MODELS
            '''
            del embedding, attention, clf, decoder

            '''covariate embedding'''
            embedding = Embedding(ncov, cts_var, cts_idx, cat_var, cat_idx, x_landmarks, x_levels, m=args.s)

            '''attention network'''

            attention = SelfAttention(dropout=0.1)
            '''contrastive learning'''
            # input is pair of latent Z
            clf = FDV_CL(m = args.s, ncov = ncov, t_landmarks = t_landmarks, h_dim=args.enc_dim)


            '''decoding the embedded vectors'''
            decoder = DecMLP_bin(input_size = zdim, output_size = args.nbin, h_dim=args.dec_dim)


            embedding.load_state_dict(torch.load(emb_path))
            attention.load_state_dict(torch.load(att_path))
            clf.load_state_dict(torch.load(clf_path))
            decoder.load_state_dict(torch.load(dec_path))


            # put models on device 0
            embedding.to(device)
            clf.to(device)
            attention.to(device)
            decoder.to(device)

            # define optimizer
            opt_emb = optim.Adam(embedding.parameters(), lr=0.1*args.emb_lr)
            opt_dec = optim.Adam(decoder.parameters(), lr=0.1*args.dec_lr)
            opt_clf = optim.Adam(clf.parameters(), lr=0.1*args.clf_lr)

            # define scheduler
            emb_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt_emb, mode='min', factor=0.1, patience=4, min_lr=0, verbose=True)
            dec_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt_dec, mode='min', factor=0.1, patience=4, min_lr=0, verbose=True)
            clf_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt_clf, mode='min', factor=0.1, patience=2, min_lr=0, verbose=True)

            decoder = nn.DataParallel(decoder, device_ids=device_ids)
            attention = nn.DataParallel(attention, device_ids=device_ids)
            clf = nn.DataParallel(clf, device_ids=device_ids)
            
            retrained = True
        
        if epoch - best_epoch >=10 and retrained:
            print('Model stopped due to early stopping')
            break            

    '''
    SAVE TRAINING HISTORIES
    '''

    train_hist = { "NLL": train_loss_NLL_hist, "ctra": train_loss_ctr_hist, "est": train_loss_est_hist}
    val_hist = { "NLL": valid_loss_NLL_hist, "ctra": valid_loss_ctr_hist, "est": valid_loss_est_hist}

    with open(result_path+"/" + args.store_name+"_train_hist.pkl",'wb') as f:
        pickle.dump(train_hist, f)
        pickle.dump(val_hist, f)


    # In[ ]:

    plt.figure()
    plt.plot(train_loss_NLL_hist,label="train losses")
    plt.plot(valid_loss_NLL_hist,label="valid losses")
    plt.axvline(best_epoch, color='gray',linestyle='--')

    plt.legend()

    plt.savefig(train_plot_path+'/'+args.store_name+'_training_hist'+'.png')


    # In[ ]:

    plt.figure()
    plt.plot(train_loss_ctr_hist,label="train")
    plt.plot(valid_loss_ctr_hist,label="valid")
    plt.axvline(best_epoch, color='gray',linestyle='--')

    plt.title("contrastive learning losses")
    plt.legend()
    plt.savefig(train_plot_path+'/'+args.store_name+'_training_crt_hist'+'.png')


    plt.figure()
    plt.plot(train_loss_est_hist,label="train")
    plt.plot(valid_loss_est_hist,label="valid")
    plt.axvline(best_epoch, color='gray',linestyle='--')

    plt.title('point estimation loss per epoch')

    plt.legend()
    plt.savefig(train_plot_path+'/'+args.store_name+'_training_est_hist'+'.png')

    # In[ ]:

if args.evaluate== 'True':
    '''
    LOAD THE BEST MODELS
    '''
    del embedding, attention, clf, decoder

    '''covariate embedding'''
    embedding = Embedding(ncov, cts_var, cts_idx, cat_var, cat_idx, x_landmarks, x_levels, m=args.s)

    '''attention network'''

    attention = SelfAttention(dropout=0.1)
    '''contrastive learning'''
    # input is pair of latent Z
    clf = FDV_CL(m = args.s, ncov = ncov, t_landmarks = t_landmarks, h_dim=args.enc_dim)


    '''decoding the embedded vectors'''
    decoder = DecMLP_bin(input_size = zdim, output_size = args.nbin, h_dim=args.dec_dim)


    embedding.load_state_dict(torch.load(emb_path))
    attention.load_state_dict(torch.load(att_path))
    clf.load_state_dict(torch.load(clf_path))
    decoder.load_state_dict(torch.load(dec_path))


    # put models on device 0
    embedding.to(device)
    clf.to(device)
    attention.to(device)
    decoder.to(device)


    decoder = nn.DataParallel(decoder, device_ids=device_ids)
    attention = nn.DataParallel(attention, device_ids=device_ids)
    clf = nn.DataParallel(clf, device_ids=device_ids)


    embedding.eval()
    decoder.eval()
    attention.eval()
    clf.eval()

    test_e_loss = test_c_loss= 0
    test_pred_t = []
    test_pred_raw = []
    test_t = []
    test_e = []
    test_NLL = []
    # test_x = []
    for i, (x, y, e, mask) in enumerate(v_pair1):

        if i%1000==0:
            print('   iteration '+str(i))
        x= x.to(device).float()
        y = y.to(device).float()
        e = e.to(device)
        mask = mask

        x_emb, var_list = embedding(x.float())
        del x

        # re-indexing mask
        reindex = torch.tensor([np.where(cov_list==var)[0][0] for var in var_list])
        mask_new = torch.index_select(mask, 1, reindex)
        del mask
        mask_attn = attention_mask(mask_new, ncov, p=0)
        del mask_new
        z, attn_score = attention(x_emb, mask = mask_attn)
        del mask_attn

        p_raw = decoder(clf.module.enc(z))

        loss_NLL = NLL_reg(p_raw, y, e, tt, collapsed=False)

        # weighted average
        t_wa_hat = wt_avg(p_raw, tt)

    #     test_x.append(x)
        test_pred_raw.append(p_raw.detach().cpu())
        test_pred_t.append(t_wa_hat.squeeze().detach().cpu())
        test_t.append(y.squeeze().detach().cpu())
        test_e.append(e.squeeze().detach().cpu())
        test_NLL.append(loss_NLL.detach().cpu().squeeze())

        del loss_NLL, p_raw, t_wa_hat, z, x_emb



    test_pred_raw = torch.cat(test_pred_raw)
    test_e = torch.cat(test_e)
    test_t = torch.cat(test_t)
    test_pred_t = torch.cat(test_pred_t)
    test_NLL = torch.cat(test_NLL)


    # In[ ]:

    plt.figure()
    attn_score_avg = attn_score.mean(axis=0)
    ax = sns.heatmap(attn_score_avg.detach().cpu().numpy())
    plt.xticks_l = list(cov_list)
    plt.title('Attention Score')
    plt.savefig(plot_path+'/'+args.store_name+'_attention_wt'+'.png')


    val_result = { "t": test_t, "e": test_e, "pred_raw": test_pred_raw, 'pred_t': test_pred_t, 'NLL': test_NLL, 'tt':tt.detach().cpu()}
    pickle.dump( val_result, open( result_path+"/"+args.store_name+"_valid_result.pkl", "wb" ) )

    del val_result

    plt.figure()
    sns.displot(test_NLL.numpy())
    test_NLL.mean()
    plt.savefig(plot_path+'/'+args.store_name+'_likelihood'+'.png')

    # In[ ]:


    # In[ ]:



    plt.figure()

    select_idx = np.random.choice(torch.where(test_e==1)[0], 500)
    sns.scatterplot(test_pred_t[select_idx].cpu().detach().numpy().squeeze(), test_t[select_idx].cpu())
    plt.plot(test_t[select_idx].cpu(), test_t[select_idx].cpu())
    plt.xlabel('Predicted')
    plt.ylabel('True Event')

    test_CI = get_CI_raw(test_e, test_t, test_pred_t)[0]
    test_CI
    plt.title(args.model_name+' with CI '+str(round(test_CI,3)))

    plt.savefig(plot_path+'/'+args.store_name+'_prediction'+'.png')




