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
from transformers import MarianMTModel, MarianTokenizer
from nltk.tokenize import sent_tokenize
import re
from utils.translate import *
import nltk
from utils.eda import eda
import jieba


def translation(text, model, tokenizer):
    text = clean_text(text)
    text = translate(text, model, tokenizer)
    return text

def load_dataset(encoder=None, dataset_name='en', sample=1.0, aug=False, ae_model=None, ae_hidden=768, aug_num=0, da_model=None, aug_type=None, backtrans=False, eda_aug=False):
    # load dataset
    sample = int(sample * 100)
    dataset_path = f'datasets/{dataset_name}/'
    if backtrans:
        train_dataset = pd.read_csv(dataset_path+f'train_{sample}_backtrans.tsv', sep='\t', names=['labels', 'text'])[['text', 'labels']]
    else:
        train_dataset = pd.read_csv(dataset_path+f'train_{sample}.tsv', sep='\t', names=['labels', 'text'])[['text', 'labels']]
        if eda_aug:
            aug_data = {'text':[], 'labels':[]}
            for i in range(len(train_dataset)):
                text = train_dataset.iloc[i]['text']
                label = train_dataset.iloc[i]['labels']
                if dataset_name != 'en':
                    if dataset_name == 'cn':
                        text_jb = jieba.lcut(text)
                        text = " ".join(text_jb)
                    # for indonesian and chinese
                    eda_text = eda(text, alpha_sr=0, alpha_ri=0)
                else:
                    # for english
                    eda_text = eda(text)
                for j in eda_text:
                    aug_data['text'].append(j)
                    aug_data['labels'].append(label)
            aug_dataset = pd.DataFrame(aug_data)
            train_dataset = train_dataset.append(aug_dataset, ignore_index=True)
    
    val_dataset = pd.read_csv(dataset_path+'dev.tsv', sep='\t', names=['labels', 'text'])[['text', 'labels']]
    test_dataset = pd.read_csv(dataset_path+'test.tsv', sep='\t', names=['labels', 'text'])[['text', 'labels']]

    # train_dataset = train_dataset.sample(100)

    # encode the text into sentence embedding
    train_enc= encoder.encode(train_dataset.text.values, convert_to_tensor=True)
    val_enc = encoder.encode(val_dataset.text.values, convert_to_tensor=True)
    test_enc = encoder.encode(test_dataset.text.values, convert_to_tensor=True)

    # get labels
    train_labels = train_dataset.labels.values
    val_labels = val_dataset.labels.values
    test_labels = test_dataset.labels.values

    # if backtrans:
    #     train_labels = np.concatenate((train_labels, train_labels))

    # train_enc = train_enc[:5]

    if aug:
        # Linear Transformation
        if aug_type == 'linear':
            train_aug_lin = linear(train_enc, aug_num)
            train_enc = torch.cat((train_enc, train_aug_lin), 0)
            train_labels = np.concatenate((train_labels, train_labels))
        elif aug_type == 'ae':
            train_aug_ae = autoencoder(train_enc, ae_model, ae_hidden)
            train_enc = torch.cat((train_enc, train_aug_ae), 0)
            train_labels = np.concatenate((train_labels, train_labels))
        elif aug_type == 'da':
            train_aug_da = denoising_ae(train_enc, da_model, ae_hidden)
            train_enc = torch.cat((train_enc, train_aug_da), 0)
            train_labels = np.concatenate((train_labels, train_labels))
        elif aug_type == 'all':
            train_aug_lin = linear(train_enc, aug_num)
            train_aug_ae = autoencoder(train_enc, ae_model, ae_hidden)
            train_aug_da = denoising_ae(train_enc, da_model, ae_hidden)
            train_enc = torch.cat((train_enc, train_aug_lin, train_aug_ae, train_aug_da), 0)
            train_labels = np.concatenate((train_labels, train_labels, train_labels, train_labels))
        elif aug_type == 'linear_ae':
            train_aug_lin = linear(train_enc, aug_num)
            train_aug_ae = autoencoder(train_enc, ae_model, ae_hidden)
            train_enc = torch.cat((train_enc, train_aug_lin, train_aug_ae), 0)
            train_labels = np.concatenate((train_labels, train_labels, train_labels))
        elif aug_type == 'linear_da':
            train_aug_lin = linear(train_enc, aug_num)
            train_aug_da = denoising_ae(train_enc, da_model, ae_hidden)
            train_enc = torch.cat((train_enc, train_aug_lin, train_aug_da), 0)
            train_labels = np.concatenate((train_labels, train_labels, train_labels))
        elif aug_type == 'ae_da':
            train_aug_ae = autoencoder(train_enc, ae_model, ae_hidden)
            train_aug_da = denoising_ae(train_enc, da_model, ae_hidden)
            train_enc = torch.cat((train_enc, train_aug_ae, train_aug_da), 0)
            train_labels = np.concatenate((train_labels, train_labels, train_labels))
        elif aug_type == 'linlin':
            train_aug_lin = linear(train_enc, aug_num)
            train_noise = noise(train_enc)
            train_aug_ae = autoencoder(noise(train_enc), ae_model, ae_hidden)
            train_aug_da = denoising_ae(noise(train_enc), da_model, ae_hidden)            
            train_all = torch.cat((train_aug_ae, train_aug_da), 0)
            train_mean = torch.mean(train_all, 0).unsqueeze(0) * train_noise # best
            # train_mean = torch.mean(train_all, 0).unsqueeze(0)
            train_enc = torch.cat((train_enc, train_mean), 0)
            train_labels = np.concatenate((train_labels, train_labels))
        elif aug_type == 'try':
            train_aug = noise(train_enc) + train_enc
            train_enc = torch.cat((train_enc, train_aug), 0)
            train_labels = np.concatenate((train_labels, train_labels))
        elif aug_type == 'trytry':
            train_aug_lin = linear(train_enc, aug_num)
            train_aug_ae = autoencoder(train_enc, ae_model, ae_hidden)
            train_aug_da = denoising_ae(train_enc, da_model, ae_hidden)
            # train_all = torch.cat((train_aug_ae, train_aug_da, train_aug_lin), 0)
            train_all = torch.stack([train_aug_da, train_aug_ae, train_aug_lin])
            train_mean = torch.mean(train_all, 0)#.unsqueeze(0) #* train_enc
            


            # print('mean shape: ',train_mean.shape)
            # print(train_aug_da.shape)
            # print(train_enc.shape)
            # train_mean = noise(train_enc) * train_aug_lin
            # sim_lin = util.pytorch_cos_sim(train_enc, train_aug_lin)
            # sim_ae = util.pytorch_cos_sim(train_enc, train_aug_ae)
            # sim_da = util.pytorch_cos_sim(train_enc, train_aug_da)
            # sim_mean = util.pytorch_cos_sim(train_enc, train_mean)
            # print('ae model: ', ae_model)
            # print('sim lin: ',sim_lin)
            # print('sim ae: ',sim_ae)
            # print('sim da: ', sim_da)
            # print('sim mean: ',sim_mean)
            
            train_enc = torch.cat((train_enc, train_mean), 0)
            train_labels = np.concatenate((train_labels, train_labels))
            # print(train_enc.shape)
            # exit()
        else:
            print('wrong aug_type')
            exit()
    
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

def linear(embedding, aug_num):
    augmented = embedding + aug_num
    return augmented

def denoising_ae(embedding, model, hidden):
    da = AutoEncoder.load_from_checkpoint(
        f'ae/best/ae-quora-den-{model}.ckpt', embedding_dim=768, hidden_dim=hidden, lr=1e-4
    ).cuda()
    # train_noise = add_noise(embedding, 0.5, 'gaussian')
    train_noise = embedding
    augmented = da(train_noise).to('cuda').detach()
    return augmented

def autoencoder(embedding, model, hidden):
    ae = AutoEncoder.load_from_checkpoint(
        f'ae/best/ae-quora-den-{model}.ckpt', embedding_dim=768, hidden_dim=hidden, lr=1e-4
    ).cuda()
    # train_noise = add_noise(embedding, 0.5, 'gaussian')
    train_noise = embedding
    augmented = ae(train_noise).to('cuda').detach()
    return augmented

def noise(embedding):
    mean_emb = torch.mean(embedding).cuda()
    std_emb = torch.std(embedding).cuda() #+ 10
    noise = torch.normal(mean_emb, std_emb, size=(1,768)).cuda()  
    return embedding * noise

def deletion(embedding, frac):
    for i in range(len(embedding)):
        length = len(embedding[i])
        r = np.random.randint(0,length,int(length*frac))
        # print(r)
        for j in r:
            embedding[i][j] = 0.
    return embedding