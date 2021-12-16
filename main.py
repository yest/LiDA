from argparse import ArgumentParser
from utils.my_datamodule import MyDataModule
from utils import transform
from utils.dataset import load_dataset
from models.bilstm import BiLSTM
from sentence_transformers import SentenceTransformer, util
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities import seed
import random
import pandas as pd
from ae.ae import AutoEncoder
import torch
from torch.nn.functional import normalize
import numpy as np
from sklearn.model_selection import train_test_split
import wandb
from pytorch_lightning.loggers import WandbLogger

import time


def main(hparams):
    seed.seed_everything(1)
    wandb.login(host='http://localhost:8081')
    project_name = hparams.dataset if hparams.project_name == None else hparams.project_name
    run = wandb.init(
        project = project_name,
        save_code=True
        # project=hparams.dataset,
        # project=hparams.dataset+'_denoising',
        # project=f'{hparams.dataset}_{hparams.lr}'
        # entity='denritchie'
    )   
    wandb_logger = WandbLogger()
    wandb.config.update(hparams)

    wandb.save('experiment_new.py')
    wandb.save('main_new.py')
    wandb.save('utils/dataset_new.py')
    wandb.save('models/bilstm.py')

    print(f'Augmenting: {hparams.augmenting}')
    encoder = SentenceTransformer(hparams.sbert)

    start = time.time()
    datasets = load_dataset(
        encoder, 
        hparams.dataset, 
        hparams.sample, 
        hparams.augmenting, 
        hparams.ae_model, 
        hparams.ae_hidden, 
        hparams.aug_number, 
        hparams.da_model, 
        hparams.aug_type, 
        hparams.backtrans,
        hparams.eda_aug)
    end = time.time()
    print(end-start)
    # exit()

    print(f"Train data: {len(datasets['train']['text'])}")
    num_labels = len(set(datasets['train']['labels']))
    print(num_labels)
    dm = MyDataModule(datasets, hparams.batch_size)
    model = BiLSTM(768, hparams.hidden_dim, num_labels, hparams.dropout, hparams.lr)

    filename = wandb.run.name
    checkpoint_callback = ModelCheckpoint(
        dirpath='saved/',
        monitor='val_mcc',
        mode='max',
        # filename='test')
        filename=filename
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=1,
        max_epochs=hparams.epochs,
        deterministic=True,
        callbacks=[checkpoint_callback, lr_monitor
            # EarlyStopping(monitor='val_f1',
            #               patience=hparams.patience, 
            #               mode='max'),
            
        ], 
        auto_lr_find=True,
        # check_val_every_n_epoch=5,
        # min_epochs = 100,
    )
    # trainer.tune(model,dm)
    trainer.fit(model, dm)
    trainer.test(datamodule=dm)
    print(checkpoint_callback.best_model_score)

    run.finish()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--sbert', type=str, default='stsb-xlm-r-multilingual')#paraphrase-xlm-r-multilingual-v1 / stsb-xlm-r-multilingual
    parser.add_argument('--model', type=str, default='bilstm')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--aug_number', type=float, default=0.0)
    parser.add_argument('--augmenting', type=bool, default=False)
    parser.add_argument('--hidden_dim', type=int, default=500)
    parser.add_argument('--dataset', type=str, default='en')
    parser.add_argument('--sample', type=float, default=1.0)
    parser.add_argument('--ae_model', type=str, default=None)
    parser.add_argument('--da_model', type=str, default=None)
    parser.add_argument('--ae_hidden', type=int, default=None)
    parser.add_argument('--project_name', type=str, default=None)
    parser.add_argument('--aug_type', type=str, default=None)
    parser.add_argument('--backtrans', type=bool, default=False)
    parser.add_argument('--eda_aug', type=bool, default=False)
    args = parser.parse_args()

    main(args)