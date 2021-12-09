import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
# from ae.ae_lstm import add_noise, AutoEncoder
from ae.ae import add_noise, AutoEncoder
import torch
from sentence_transformers import SentenceTransformer, util
import pytorch_lightning as pl
import torch.nn as nn


# class AutoEncoder(pl.LightningModule):
#     def __init__(self, embedding_dim, hidden_dim, lr):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(embedding_dim, hidden_dim),
#             # nn.ReLU(),
#             nn.Linear(hidden_dim,hidden_dim),            
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(hidden_dim,hidden_dim),
#             # nn.ReLU(),
#             nn.Linear(hidden_dim, embedding_dim),
            
#         )
        
#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x


def load_dataset(encoder=None, dataset_name='en', sample=1.0, aug=False, ae_model=None, ae_hidden=768, aug_num=0):
    # load dataset
    sample = int(sample * 100)
    dataset_path = f'datasets/{dataset_name}/'
    train_dataset = pd.read_csv(dataset_path+f'train_{sample}.tsv', sep='\t', names=['labels', 'text'])[['text', 'labels']]
    val_dataset = pd.read_csv(dataset_path+'dev.tsv', sep='\t', names=['labels', 'text'])[['text', 'labels']]
    test_dataset = pd.read_csv(dataset_path+'test.tsv', sep='\t', names=['labels', 'text'])[['text', 'labels']]

    # get sample from train dataset
    # if sample: 
    #     train_dataset = pd.read_csv(dataset_path+f'train_{sample}.tsv', sep='\t', names=['labels', 'text'])[['text', 'labels']]
        # train_dataset = train_dataset.sample(frac=sample, random_state=1)
    
    # encode the text into sentence embedding
    train_enc= encoder.encode(train_dataset.text.values, convert_to_tensor=True)
    val_enc = encoder.encode(val_dataset.text.values, convert_to_tensor=True)
    test_enc = encoder.encode(test_dataset.text.values, convert_to_tensor=True)

    # print(train_enc.shape)

    # get labels
    train_labels = train_dataset.labels.values
    val_labels = val_dataset.labels.values
    test_labels = test_dataset.labels.values

    if(aug):
        # type 1 augmentation: linear transformation
        if aug_num > 0:
            train_aug = train_enc + aug_num
        else:

            # type 2 augmentation: autoencoder
            ae = AutoEncoder.load_from_checkpoint(
                # 'ae/saved/ae-quora-den-decent-sun-78.ckpt', embedding_dim=768, hidden_dim=768, lr=5e-4 # best so far ae denoising no relu
                # 'ae/saved/ae-quora-den-prime-shape-124.ckpt', embedding_dim=768, hidden_dim=768, lr=1e-4
                # 'ae/saved/ae-quora-den-worldly-cloud-133.ckpt', embedding_dim=768, hidden_dim=768, lr=1e-4
                
                # 'ae/saved/ae-quora-den-silvery-tree-79.ckpt', embedding_dim=768, hidden_dim=768, lr=5e-4 # best 2 so far ae vanilla no relu
                # 'ae/saved/ae-quora-den-copper-microwave-90.ckpt', embedding_dim=768, hidden_dim=768, lr=5e-4 # best so far ae vanilla no relu
                
                # 'ae/saved/ae-quora-den-restful-rain-90.ckpt', embedding_dim=768, hidden_dim=768, lr=1e-4 #1

                # f'ae/best/ae-quora-den-{ae_model}.ckpt', embedding_dim=768, hidden_dim=ae_hidden, lr=1e-4
                f'ae/saved/ae-quora-den-{ae_model}.ckpt', embedding_dim=768, hidden_dim=ae_hidden, lr=1e-4


                # 'ae/saved/ae-quora-lemon-smoke-12-epoch=194.ckpt', embedding_dim=768, hidden_dim=768, lr=3e-5
                # 'ae/saved/ae-quora-valiant-thunder-21-epoch=34.ckpt', embedding_dim=768, hidden_dim=768, lr=5e-4
            ).cuda()
            # create synthetic data using autoencoder

            # add noise
            # train_noise = add_noise(train_enc, 0.5, 'gaussian')
            # train_aug = ae(train_noise.cpu()).to('cuda').detach()

            # no noise
            train_aug = ae(train_enc).to('cuda').detach()

            # add noise to augmented embedding
            # train_aug = add_noise(train_aug, 0.5, 'gaussian')

        # concatenate original train and augmentedf train
        train_enc = torch.cat((train_enc, train_aug), 0)
        # train_enc = torch.cat((train_aug, train_enc), 0)
        train_labels = np.concatenate((train_labels, train_labels))
    
    return {
        'train': {
            'text': train_enc,
            'labels': train_labels
        },
        'val': {
            'text': val_enc,
            'labels': val_labels
        },
        'test': {
            'text': test_enc,
            'labels': test_labels
        }
    }