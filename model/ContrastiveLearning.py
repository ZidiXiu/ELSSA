
import torch.nn as nn
import torch.nn.functional as F
import torch

import torch
from torch import nn, optim
import numpy as np

class FDV_CL(nn.Module):
    def __init__(self, m, ncov, h_dim=0, t_landmarks = None, train_time = None, percentile=False, tau = 1.0):
        super(FDV_CL, self).__init__()
        # learnable temperature
        self.log_tau = torch.nn.Parameter(torch.Tensor([np.log(tau)]))
        self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.m = m
        self.ncov = ncov
        self.time_landmark, self.time_emb_landmark = self.time_emb_init(t_landmarks, train_time, percentile)
        
        if h_dim[0] == 0:
            self.enc = nn.Identity()
        else:
            net = []
            hs = [m] + h_dim + [m]
            for h0, h1 in zip(hs, hs[1:]):
                net.extend([
                    nn.Linear(h0, h1),
                    nn.ReLU(),
                ])
            net.pop()  # pop the last ReLU for the output layer
            self.enc = nn.Sequential(*net)
            
        
    def forward(self, z, t, e):
        m = self.m
        tau = torch.sqrt(torch.exp(self.log_tau))    
        device = z.device
                
        batch_dim = z.size(0)
        # by dimension encoding
        # [batch_size, n_emb]
        hz = (self.norm(self.enc(z)))/tau
           
        # [batch_size, n_emb]
        hy = self.norm((self.linear_interpolation_time(t, e)))/tau
                
#         hy_new = torch.repeat_interleave(self.linear_interpolation_time(t, e)/tau, torch.tensor(batch_dim*[self.ncov]).to(device), dim=0).view(-1,hz.size()[-1])
                
        del z, t, e
        
        # [batch_size, batch_size]
        similarity_matrix = hz @ hy.t()
                
        del hz, hy
        pos_mask = torch.eye(batch_dim,dtype=torch.bool)
        
        g = similarity_matrix[pos_mask].view(batch_dim,-1)
        g0 = similarity_matrix[~pos_mask].view(batch_dim,-1)
            
        del pos_mask
        logits = g0 - g
            
        slogits = torch.logsumexp(logits,1).view(-1,1)
            
        labels = torch.tensor(range(batch_dim),dtype=torch.int64).to(device)
        dummy_ce = self.criterion(similarity_matrix,labels) - torch.log(torch.Tensor([batch_dim]).to(device))
                
        del similarity_matrix
        dummy_ce = dummy_ce.view(-1,1)
            
        output = dummy_ce.detach()+torch.exp(slogits-slogits.detach())-1
        del dummy_ce
        output = torch.clamp(output,-5,15)
        
        return output
    
    
    def linear_interpolation_time(self, batch_t, batch_e):
        '''
        Time embedding based on event type
        '''
        # written for torch
        # returns index
        # a[i-1] < v <= a[i]
        indx = torch.searchsorted(self.time_landmark.to(batch_t.device), batch_t)
        # combine the first two indices, and the last two indices
        # to include the unobserved minimum and maximum value
        # combines [0,1], and [m-1, m]
        indx = torch.where(indx==0, 1, indx)
        indx = torch.where(indx==self.m, self.m-1, indx)

        
#         print(batch_t.size(), batch_e.size(), len(indx))
        
        # for event
        coef = (self.time_emb_landmark[indx] - self.time_emb_landmark[indx-1])/(self.time_landmark[indx].to(batch_t.device) - self.time_landmark[indx-1].to(batch_t.device)).view(len(indx),1)
#         print(coef.size())
        out = self.time_emb_landmark[indx-1] + coef*(batch_t-self.time_landmark[indx-1].to(batch_t.device)).view(len(indx),1)    # linear interpolation

        # for censoring
        censor_mask = torch.vstack([(torch.cat((torch.zeros(idx), torch.ones(self.m-idx))).view(1, self.m))/(self.m-idx) for idx in indx.to('cpu')])
        out_censor = torch.matmul(censor_mask.to(batch_t.device), self.time_emb_landmark.T)
        
        del censor_mask, coef

        return (out*batch_e.view(len(out),1) + out_censor*(1-batch_e).view(len(out),1)).float()
    
    def time_emb_init(self, t_landmarks, train_time, percentile = True):
        '''
        embedding of time-to-event distribution 

        '''

        if type(t_landmarks) == type(None):
            if percentile:
                '''based on each covariate percentile'''
                t_landmarks = torch.tensor(np.percentile(train_time, np.linspace(0,100,self.m))).contiguous()
            else:
                '''based on the each covariate range'''
                t_landmarks = torch.tensor(np.linspace(np.min(train_time),np.max(train_time),self.m, endpoint=True)).contiguous()

        # save the initialized landmarks as a torch parameter dictionary

        t_emb_landmarks = torch.eye(self.m)

        return t_landmarks, torch.nn.Parameter(t_emb_landmarks)    
    
    def norm(self,z):
        return torch.nn.functional.normalize(z,dim=-1)