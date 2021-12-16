from sentence_transformers import SentenceTransformer, util

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim

from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import time
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import pytorch_lightning as pl
from pytorch_lightning.utilities import seed
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import torchmetrics
from torchmetrics import F1

import warnings
warnings.filterwarnings('ignore')

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


class MyDataset(Dataset):
    def __init__(self, text1, text2):
        self.text1 = text1
        self.text2 = text2

    def __len__(self):
        return len(self.text1)

    def __getitem__(self, idx):
        text1 = self.text1[idx]
        text2 = self.text2[idx]
        return {'text1': text1, 'text2': text2}


class MyDataModule(pl.LightningDataModule):
    def __init__(self, datasets, encoder, batch_size, denoising=False):
        super().__init__()

        self.train1 = datasets['train1'].cuda()
        self.train2 = datasets['train2'].cuda()        
        self.val1 = datasets['val1'].cuda()
        self.val2 = datasets['val2'].cuda()

        if denoising:
            self.train1 = add_noise(self.train1, 0.5, 'gaussian')
            self.train2 = self.train1
            # self.val1 = add_noise(self.val1, 0.5, 'gaussian')
            self.val2 = self.val1

        # df_train = pd.read_csv(self.train).fillna("none")
        # df_val = pd.read_csv(self.val).fillna("none")

        # df_train.columns = ['text','label']
        # df_val.columns = ['text','label']
        
        # train_text = encoder.encode(df_train.text.values, convert_to_tensor=True)
        # val_text = encoder.encode(df_val.text.values, convert_to_tensor=True)


        self.train_dataset = MyDataset(
            text1 = self.train1,
            text2 = self.train2,
        )
        self.val_dataset = MyDataset(
            text1 = self.val1,
            text2 = self.val2,
        )
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
    
    # def add_noise(self, emb, frac, noise_type='zero'):
    #     length = len(emb)
    #     if noise_type == 'zero':
    #         r = np.random.randint(0,length,int(length*frac))
    #         for i in r:
    #             emb[i] = 0.
    #         # emb[i] = random.random()
    #     elif noise_type == 'gaussian':
    #         mean_emb = torch.mean(emb)
    #         std_emb = torch.std(emb)
    #         noise = torch.normal(mean_emb, std_emb, size=(1,768)).cuda()
    #         emb = emb + noise
    #     return emb



class AutoEncoder(pl.LightningModule):
    def __init__(self, embedding_dim, hidden_dim, lr):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            # nn.ReLU(),
            # nn.Linear(hidden_dim,150),

            # nn.Dropout(0.2),
            # nn.ReLU()

            # nn.Tanh(),
            # nn.Linear(hidden_dim, 150),
            
        )
        self.decoder = nn.Sequential(
            # nn.Linear(150, hidden_dim),
            # nn.ReLU(),

            # nn.Linear(150,hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
            
            
            # nn.Dropout(0.2),
            
        )
        self.lr = lr

        self.save_hyperparameters()
        
    def forward(self, x):
        # x = self.denoising(x, 0.3)
        # x = x.to('cpu')
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x = batch['text1']
        y = batch['text2']
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['text1']
        y = batch['text2']
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.8)
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': {
            #     'scheduler': scheduler,
            #     'monitor': 'val_loss'
            # }
        }
        # return optimizer
    
    # def denoising(self, emb, frac):
    #     length = len(emb)
    #     r = np.random.randint(0,length,int(length*frac))
    #     for i in r:
    #         emb[i] = 0.
    #     return emb

def add_noise(emb, frac, noise_type='zero'):
    length = len(emb)
    if noise_type == 'zero':
        r = np.random.randint(0,length,int(length*frac))
        for i in r:
            emb[i] = 0.
        # emb[i] = random.random()
    elif noise_type == 'gaussian':
        mean_emb = torch.mean(emb).cuda()
        std_emb = torch.std(emb).cuda() + 2
        # std_emb.to('cuda')
        noise = torch.normal(mean_emb, std_emb, size=(1,768)).cuda()  
        # noise2 = torch.normal(mean_emb, std_emb, size=(1,768)).cuda()
        # noise = torch.distributions.LogNormal(mean_emb, std_emb).sample((1,768))*mean_emb

        emb = emb + noise
        # emb = noise

        # r = np.random.randint(0,length,int(length*0.5))
        # for i in r:
        #     emb[i] = 0.
    return emb.cuda()