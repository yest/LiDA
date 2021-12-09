from utils.my_dataset import MyDataset
from torch.utils.data import DataLoader
import pandas as pd
import pytorch_lightning as pl
import torch
import numpy as np
import random

class MyDataModule(pl.LightningDataModule):
    def __init__(self, datasets, batch_size):
        super().__init__()
        self.batch_size = batch_size            
            
        self.train_dataset = MyDataset(
            text = datasets['train']['text'],
            label= datasets['train']['labels']
        )

        self.valid_dataset = MyDataset(
            text = datasets['val']['text'],
            label= datasets['val']['labels']
        )

        self.test_dataset = MyDataset(
            text = datasets['test']['text'],
            label= datasets['test']['labels']
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            # drop_last=True,
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