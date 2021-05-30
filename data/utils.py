import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
import torch
import pandas


# for training/testing/validation split
def formatted_data(x, t, e, idx):
    death_time = np.array(t[idx], dtype=float)
    censoring = np.array(e[idx], dtype=float)
    covariates = np.array(x[idx])

    print("observed fold:{}".format(sum(e[idx]) / len(e[idx])))
    survival_data = {'x': covariates, 't': death_time, 'e': censoring}
    return survival_data


class SimpleDataset(Dataset):
    def __init__(self, x, y, e, transform=False, mean=0, std = 1):
        self.data = x
        self.targets = y
        self.label = e
        self.transform = transform
        
        if self.transform:
            self.mean, self.std = mean, std
            
    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]
        label = self.label[index]
        
        if self.transform:
            return (img-self.mean)/self.std, target, label
        else:
            return img, target, label
        
        
    def __len__(self):
        return len(self.data)

class SimpleDataset_masked(Dataset):
    def __init__(self, x, y, e, mask, transform=False, mean=0, std = 1):
        self.data = x
        self.targets = y
        self.label = e
        self.mask = mask
        self.transform = transform
        
        if self.transform:
            self.mean, self.std = mean, std
            
    def __getitem__(self, index):
        img = self.data[index]
        target = self.targets[index]
        label = self.label[index]
        mask = self.mask[index]
        
        if self.transform:
            return (img-self.mean)/self.std, target, label
        else:
            return img, target, label, mask
        
        
    def __len__(self):
        return len(self.data)
    
    

# class SimpleDataset_trans(Dataset):
#     def __init__(self, x, y, e, transform=False, covlist = None, mean=0, std = 1):
#         self.data = x
#         self.targets = y
#         self.label = e
#         self.transform = transform
#         self.covlist = covlist
#         if not self.covlist:
#             self.covlist = np.arange(self.data.shape[1])
        
#         if self.transform:
#             self.mean, self.std = mean, std
            
#     def __getitem__(self, index):
#         img = self.data[index]
#         target = self.targets[index]
#         label = self.label[index]
        
#         if self.transform:
#             return (img[self.covlist]-self.mean)/self.std, target, label
#         else:
#             return img, target, label
        
        

#     def __len__(self):
#         return len(self.data)        
        

#     def __len__(self):
#         return len(self.data)


### Cleaning up
# one-hot-encoding all categorical variables
def one_hot_encoder(data, encode):
    data_encoded = data.copy()
    encoded = pandas.get_dummies(data_encoded, prefix=encode, columns=encode)
#    print("head of data:{}, data shape:{}".format(data_encoded.head(), data_encoded.shape))
#    print("Encoded:{}, one_hot:{}{}".format(encode, encoded.shape, encoded[0:5]))
    return encoded