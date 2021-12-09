import opcode
from sentence_transformers import SentenceTransformer, util

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

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

from utils.my_datamodule import MyDataModule


class BiLSTM(pl.LightningModule):
    def __init__(self, embedding_dim, hidden_dim, num_labels, dropout, lr):
        super().__init__()

        self.lr = lr
        self.num_labels = num_labels
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, bidirectional=False, dropout=dropout)
        self.fc = nn.Linear(hidden_dim*2, num_labels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU()
        # self.softmax = nn.Softmax(dim=1)

        # self.automatic_optimization = False

        self.save_hyperparameters()
    
    def forward(self, x):
        # x = torch.Tensor([x,])
        x = x.unsqueeze(0)
        embedded = self.dropout(x)
        output, (hidden, cell) = self.rnn(embedded)
        hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        return self.fc(self.relu(hidden.squeeze(0)))  
        # return self.fc(hidden.squeeze(0))       
    
    def training_step(self, batch, batch_idx):
        x = batch['text']
        y = batch['label']
        y_hat = self(x)
        # print(x.shape)
        # print(y_hat.shape)
        loss = F.cross_entropy(y_hat, y)
        # print(loss)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch['text']
        y = batch['label']
        y_hat = self(x)
        
        loss = F.cross_entropy(y_hat, y)
        y_hat = torch.argmax(y_hat, dim=-1)#.float()
        result = {'y':y, 'y_hat':y_hat}
        self.log('val_loss', loss)
        return result
    
    def validation_epoch_end(self, outputs):
        y = []
        y_hat = []
        for out in outputs:
            y.extend(out['y'].long().cpu().detach().numpy().tolist())
            y_hat.extend(out['y_hat'].long().cpu().detach().numpy().tolist())
        val_acc = metrics.accuracy_score(y, y_hat)
        val_f1 = metrics.f1_score(
            y, y_hat, average='macro' if self.num_labels > 2 else 'binary'
        )
        val_mcc = metrics.matthews_corrcoef(y, y_hat)
        # self.logger.experiment.log({'val_f1': val_f1, 'val_acc': val_acc})
        self.log('val_acc', val_acc)
        self.log('val_f1', val_f1)
        self.log('val_mcc', val_mcc)
        # return {'val_loss': avg_loss}
    
    def test_step(self, batch, batch_idx):
        x = batch['text']
        y = batch['label']
        y_hat = self(x)
        y_hat = torch.argmax(y_hat, dim=-1)#.float()
        result = {'y':y, 'y_hat':y_hat}
        return result
    
    def test_epoch_end(self, outputs):
        y = []
        y_hat = []
        for out in outputs:
            y.extend(out['y'].long().cpu().detach().numpy().tolist())
            y_hat.extend(out['y_hat'].long().cpu().detach().numpy().tolist())
        test_acc = metrics.accuracy_score(y, y_hat)
        test_f1 = metrics.f1_score(
            y, y_hat, average='macro' if self.num_labels > 2 else 'binary'
        )
        test_mcc = metrics.matthews_corrcoef(y, y_hat)
        self.log('test_f1', test_f1)
        self.log('test_acc', test_acc)
        self.log('test_mcc', test_mcc)
        # print(metrics.precision_recall_fscore_support(y, y_hat, average='macro'))

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        # optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.8)
        # scheduler = StepLR(optimizer, step_size=5, gamma=0.9)
        return {
            'optimizer': optimizer,
            # 'lr_scheduler': {
            #     'scheduler': scheduler,
            #     'monitor': 'val_loss'
            # }
        }
        # [optimizer], [scheduler]
        # return {
        #     'optimizer': optimizer,
        #     'lr_scheduler': {
        #         'scheduler': StepLR(optimizer,step_size=100,gamma=0.1)

        #     }
        # }