# Consider cox-ph settings with exponential hazard function, then $T = \frac{-\log U}{\lambda \exp(\beta'x)}$

import math
import os


import numpy as np
import pandas

# generate simulation based on given covariates
def simulation_cox_weibull(X, coef_all, n_samples=10000, lambda_=0.5, nu_=1/100,cut_bound=1, seed=None):
    # linear relationship
    # n: number of patients        
    # the problem with the left tail is about this U!
    if seed is not None:
        np.random.seed(seed)
        U = np.random.uniform(size=n_samples)
    else:
        U = np.random.uniform(size=n_samples)
    
    # generate T accordingly
    T = (-np.log(U)/(lambda_*np.exp(np.dot(X,coef_all))))**(1/nu_)

    event = T<cut_bound
    
    return({"e":event*1, "x":X, "t": T})


# simulation of both covariates and coefficients
def simulation_cox_weibull_all(n=10000, p=3, pc=1, pval=[1/4.]*4, lambda_=7e-8, alpha_=0.2138, censor_bound=68, seed=123):
    # linear relationship
    # n: number of patients
    # p: number of total covariates
    # pc: number of categorical variables
    # pc_level: levels of categorical variable
    # pval: probabilities for each level
    # lambda_exp: parameters for baseline hazards
    
    np.random.seed(seed)
    # generate based on Bender's paper
    beta_cts = np.array([0.15,0.001, 0.10])
    X_age = np.random.normal(loc=24.3, scale = 8.38, size=n).reshape((n,1))
    X_randon = np.random.normal(loc=266.84, scale = 507.82, size=n).reshape((n,1))
    # add an interaction term
    X_int = (X_age * X_randon/3000).reshape((n,1))
    
    X_cts = np.concatenate((X_age, X_randon, X_int), axis=1)
    
    pc_level = len(pval)
#     p_total = (p-pc) + pc*pc_level
#     cat0 = np.vstack([np.random.multinomial(1, [1/4.]*4, size=n) for i in range(4)])
#     cat1 = np.stack([np.random.multinomial(1, [1/100.]*100, size=n) for i in range(100)], axis=0).reshape(n, 100)
    cat0 = np.random.multinomial(1, [1/4.]*4, size=n)
    cat1 = np.random.multinomial(1, [1/100.]*100, size=n)
    X_cat = np.concatenate((cat0, cat1), axis=1)
    num_cat_coef = X_cat.shape[1]
    beta_cat0 = np.random.uniform(0,5,cat0.shape[1])
    beta_cat1 = np.random.uniform(-0.5,0.5,cat1.shape[1])
        
    X = np.concatenate((X_cts, X_cat), axis=1)

    
    beta_linear = np.concatenate((beta_cts, beta_cat0, beta_cat1))
    # the problem with the left tail is about this U!
    U = np.random.uniform(size=n)
    # generate T accordingly
    T = (1/alpha_)*np.log(1-alpha_*np.log(U)/(lambda_*np.exp(np.dot(X,beta_linear))))
#     event = T<cut_bound
    if censor_bound>0:
        sidx = np.argsort(T,axis=0)
        TS = T[sidx]
        XS = X[sidx,:]
        np.random.seed(seed)
        # change the way censoring is defined
        # first set the maximum censoring at: censor_bound
    #    C = np.repeat(censor_bound,n)
        right_truncate = T<censor_bound
        EPS = 0
        # define C only for the right truncated samples
        C = np.random.uniform(0+EPS,censor_bound,size=len(T[right_truncate]))
        CS = np.concatenate([C,np.repeat(censor_bound,n-len(T[right_truncate]))])
        event = 1*(TS<CS)
        nonevent = CS<TS
        # observed time
        YS = TS.copy()
        YS[nonevent] = CS[nonevent]

        # shuffle back to unsorted order
        perm_idx = np.random.permutation(n)
        X = XS[perm_idx,:]
        Y = YS[perm_idx]
        event = event[perm_idx]
        C = CS[perm_idx]
        T = TS[perm_idx]
    else:
        Y = T.copy()
        C = 0
        event = np.ones(n)
        
#     print(beta_linear)
    return({"t": Y, "e":event, "x":np.concatenate((X_age, X_randon, X_cat), axis=1), "T": T,"C":C, 'cts_idx':np.arange(2), 'ohe':[cat0.shape[1], cat1.shape[1]]})


def formatted_data_simu(x, t, e, idx):
    death_time = np.array(t[idx], dtype=float)
    censoring = np.array(e[idx], dtype=float)
    covariates = np.array(x[idx])

    #print("observed fold:{}".format(sum(e[idx]) / len(e[idx])))
    survival_data = {'x': covariates, 't': death_time, 'e': censoring}
    return survival_data 

def saveDataCSV(data_dic, name, file_path):
    df_x = pandas.DataFrame(data_dic['x'])
#     one_hot_indices = data_dic['one_hot_indices']
#     # removing 1 column for each dummy variable to avoid colinearity
#     if one_hot_indices:
#         rm_cov = one_hot_indices[:,-1]
#         df_x = df_x.drop(columns=rm_cov)
    df_e = pandas.DataFrame({'e':data_dic['e']})
    df = pandas.concat([df_e, df_x], axis=1, sort=False)
    df.to_csv(file_path+'/'+name+'.csv', index=False)
    
def loadDataCSV(name, file_path):
    df = pandas.read_csv(file_path+'/'+name+'.csv')
    n_total = df.shape[1]
    z_dim = 4
    df_x = df.iloc[:,range(1,n_total)]
    df_e = df.iloc[:,0]
    return({'x':np.array(df_x), 'e':np.array(df_e)})
      
