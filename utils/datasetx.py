import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import numpy as np
from ae.ae import AutoEncoder
import torch

def load_dataset(encoder, dataset_name='sentiment', sample=None, aug=False, aug_num=0):
    if dataset_name == 'sentiment':
        dataset_path = '../../../Datasets/id/sentiment/data/'
        datasets = {
            'train': dataset_path + 'train0.csv',
            'val': dataset_path + 'dev0.csv',
            'test': dataset_path + 'test0.csv'
        }
        df_train = pd.read_csv(datasets['train']).fillna("none")
        df_val = pd.read_csv(datasets['val']).fillna("none")
        df_test = pd.read_csv(datasets['test']).fillna("none")

        df_train.columns = ['text', 'label']
        df_val.columns = ['text', 'label']
        df_test.columns = ['text', 'label']

        if sample:
            df_train = df_train.sample(frac=sample)

        train = {
            'text': encoder.encode(df_train.text.values, convert_to_tensor=True),
            'label': df_train.label.values
        }
        val = {
            'text': encoder.encode(df_val.text.values, convert_to_tensor=True),
            'label': df_val.label.values
        }
        test = {
            'text': encoder.encode(df_test.text.values, convert_to_tensor=True),
            'label': df_test.label.values
        }
    elif dataset_name == 'emotion':
        dataset_path = '../../../Datasets/id/emotion/'
        datasets = {
            'train': dataset_path + 'train_preprocess.csv',
            'test': dataset_path + 'valid_preprocess.csv'
        }
        df_train = pd.read_csv(datasets['train']).fillna("none")
        df_val = pd.read_csv(datasets['test']).fillna("none")

        df_train, df_test = train_test_split(df_train, test_size=0.1, random_state=22)

        df_train.label = df_train.label.apply(
            lambda x: ['anger', 'fear', 'happy', 'love', 'sadness'].index(x)
        )
        df_val.label = df_val.label.apply(
            lambda x: ['anger', 'fear', 'happy', 'love', 'sadness'].index(x)
        )
        df_test.label = df_test.label.apply(
            lambda x: ['anger', 'fear', 'happy', 'love', 'sadness'].index(x)
        )

        if sample:
            df_train = df_train.sample(frac=sample)

        train = {
            'text': encoder.encode(df_train.text.values, convert_to_tensor=True),
            'label': df_train.label.values
        }
        val = {
            'text': encoder.encode(df_val.text.values, convert_to_tensor=True),
            'label': df_val.label.values
        }
        test = {
            'text': encoder.encode(df_test.text.values, convert_to_tensor=True),
            'label': df_test.label.values
        }
    elif dataset_name == 'imdb':
        with open('imdb.pkl', "rb") as fIn:
            stored_data = pickle.load(fIn)
            train = {
                'text': stored_data['train_text'],
                'label': stored_data['train_labels']
            }
            val = {
                'text': stored_data['val_text'],
                'label': stored_data['val_labels']
            }
            test = {
                'text': stored_data['test_text'],
                'label': stored_data['test_labels']
            }

        if sample:
            rng = np.random.default_rng(22)
            r = rng.integers(len(train['text']), size=int(len(train['text'])*sample))
            train_text = [train['text'][i] for i in r]
            labels = [train['label'][i] for i in r]
            train = {
                'text': torch.stack(train_text),
                'label': labels
            }
        
        print(type(train['text']))
        
    if(aug):
        # ae = AutoEncoder(768, 768, 3e-5).cuda()
        ae = AutoEncoder.load_from_checkpoint(
            'ae/saved/ae-quora-den-sunny-dream-35.ckpt', embedding_dim=768, hidden_dim=300, lr=5e-4
            # 'ae/saved/ae-quora-lemon-smoke-12-epoch=194.ckpt', embedding_dim=768, hidden_dim=768, lr=3e-5
            # 'ae/saved/ae-quora-valiant-thunder-21-epoch=34.ckpt', embedding_dim=768, hidden_dim=768, lr=5e-4
            
        )
        # ae.load_state_dict(torch.load('ae/ae.bin'))
        
        train_text = train['text'].cuda()
        train_aug = ae(train_text).to('cuda').detach()
        labels = train['label']
        # train_aug = train_text + 1.00
        # train_aug2 = train_text + 0.005
        train_text = torch.cat((train_text, train_aug), 0)
        labels = np.concatenate((labels, labels))


            # print(f'train_aug: {train_aug.requires_grad}')
            # train_aug = train_aug.to('cuda')
            
            # print(f'train_text: {train_text.requires_grad}')
            

        train = {
            'text': train_text,
            'label': labels
        }

    return {
        'train': train,
        'val': val,
        'test': test,
    }#train, val, test