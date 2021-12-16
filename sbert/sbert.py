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

# import torchmetrics
# from torchmetrics import F1

# import warnings
# warnings.filterwarnings('ignore')

from transformers import (
    AutoTokenizer, 
    AutoModel,
    AdamW,
    get_linear_schedule_with_warmup,
)

from ae import AutoEncoder
# from ae import autoencoder
# /home/NCKU/2021/Research/LiDA/ae/ae.py


class BERTDataset:
    def __init__(self, text, label, tokenizer, max_len):
        self.text = text
        self.target = label
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.text)
    
    def __getitem__(self, item):
        text = str(self.text[item])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens = True,
            padding = 'max_length',
            truncation = True,
            max_length = self.max_len,
        )
        
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]

        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.target[item], dtype=torch.float),
        }


class BERTDataModule(pl.LightningDataModule):
    def __init__(self, datasets, batch_size, tokenizer, max_len):
        super().__init__()
        self.batch_size = batch_size            
            
        self.train_dataset = BERTDataset(
            text = datasets['train']['text'],
            label = datasets['train']['labels'],
            tokenizer = tokenizer,
            max_len = max_len
        )

        self.valid_dataset = BERTDataset(
            text = datasets['val']['text'],
            label = datasets['val']['labels'],
            tokenizer = tokenizer,
            max_len = max_len
        )

        self.test_dataset = BERTDataset(
            text = datasets['test']['text'],
            label= datasets['test']['labels'],
            tokenizer = tokenizer,
            max_len = max_len
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
        )


class SBERT(pl.LightningModule):
    def __init__(self, pretrained_model=None, num_labels=None, dropout=None, lr=None, aug=False, ae_model=None, ae_hidden=768, da_model=None, aug_number=0):
        super().__init__()
        self.bert = AutoModel.from_pretrained(pretrained_model, output_hidden_states=True)
        self.bert_drop = nn.Dropout(dropout)
        self.out = nn.Linear(768, num_labels) # original 768 OR cat 1536
        self.lr = lr
        self.num_labels = num_labels
        self.aug = aug
        self.ae_model = ae_model
        self.ae_hidden = ae_hidden
        self.da_model = da_model
        self.aug_number = aug_number

        self.save_hyperparameters()
    
    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] #First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask
    
    def forward(self, ids, mask, train=True):
        output = self.bert(
            ids, 
            attention_mask=mask,
        )   

        pooler = self.mean_pooling(output, mask)
        
        if self.aug:
            if train:
                ae = AutoEncoder.load_from_checkpoint(
                    # '../ae/saved/ae-quora-den-sunny-dream-35.ckpt', embedding_dim=768, hidden_dim=300, lr=5e-5
                    f'../ae/best/ae-quora-den-{self.ae_model}.ckpt', embedding_dim=768, hidden_dim=self.ae_hidden, lr=1e-4
                ).cuda()
                da = AutoEncoder.load_from_checkpoint(
                    # '../ae/saved/ae-quora-den-sunny-dream-35.ckpt', embedding_dim=768, hidden_dim=300, lr=5e-5
                    f'../ae/best/ae-quora-den-{self.da_model}.ckpt', embedding_dim=768, hidden_dim=self.ae_hidden, lr=1e-4
                ).cuda()

                train_aug_lin = pooler + self.aug_number
                train_aug_ae = ae(pooler).to('cuda').detach()
                train_aug_da = da(pooler).to('cuda').detach()
                pooler = torch.cat((pooler, train_aug_lin, train_aug_ae, train_aug_da),0)


                # newpooler = ae(pooler).to('cuda').detach()
                # pooler = torch.cat((pooler, newpooler),0)
                # newpooler = pooler + 0.15
                # pooler = torch.cat((pooler, newpooler),0)

        bo = self.bert_drop(pooler)
        output = self.out(bo)
        return output
    
    def training_step(self, batch, batch_idx):
        ids = batch["ids"].long()
        mask = batch["mask"].long()
        targets = batch["targets"].long()
        if self.aug:
            targets = torch.cat((targets,targets,targets,targets),0)
        outputs = self(
            ids=ids,
            mask=mask,
            train = True
        )
        loss = F.cross_entropy(outputs, targets)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        ids = batch["ids"]
        mask = batch["mask"]
        targets = batch["targets"].long()
        outputs = self(
            ids=ids,
            mask=mask,
            train=False
        )
        loss = F.cross_entropy(outputs, targets)
        outputs = torch.argmax(outputs, dim=-1).float()
        
        result = {'y_true':targets, 'y_pred':outputs}
        self.log('val_loss', loss)
        return result
    
    def validation_epoch_end(self, output):
        y_true = []
        y_pred = []
        for out in output:
            y_true.extend(out['y_true'].long().cpu().detach().numpy().tolist())
            y_pred.extend(out['y_pred'].long().cpu().detach().numpy().tolist())
        val_acc = metrics.accuracy_score(y_true, y_pred)
        val_f1 = metrics.f1_score(
            y_true, y_pred, average='macro' if self.num_labels > 2 else 'binary'
        )
        val_mcc = metrics.matthews_corrcoef(y_true, y_pred)
        self.log('val_acc', val_acc)
        self.log('val_f1', val_f1)
        self.log('val_mcc', val_mcc)
        # return val_f1
    
    def test_step(self, batch, batch_idx):
        ids = batch["ids"]
        mask = batch["mask"]
        targets = batch["targets"]
        outputs = self(
            ids=ids,
            mask=mask,
            train=False
        )
        outputs = torch.argmax(outputs, dim=-1).float()
        result = {'y_true':targets, 'y_pred':outputs}
#         self.log('val_loss', loss)
        return result
    
    def test_epoch_end(self, output):
        y_true = []
        y_pred = []
        for out in output:
            y_true.extend(out['y_true'].long().cpu().detach().numpy().tolist())
            y_pred.extend(out['y_pred'].long().cpu().detach().numpy().tolist())
        test_acc = metrics.accuracy_score(y_true, y_pred)
        test_f1 = metrics.f1_score(
            y_true, y_pred, average='macro' if self.num_labels > 2 else 'binary'
        )
        test_mcc = metrics.matthews_corrcoef(y_true, y_pred)
        self.log('test_f1', test_f1)
        self.log('test_acc', test_acc)
        self.log('test_mcc', test_mcc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer