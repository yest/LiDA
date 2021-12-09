from argparse import ArgumentParser
from sentence_transformers import SentenceTransformer, util
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities import seed
import random
# from datasets import load_dataset
import pickle

import wandb
from pytorch_lightning.loggers import WandbLogger
from yaml import parse


# from ae_lstm import MyDataModule, AutoEncoder
from ae import MyDataModule, AutoEncoder

def main(hparams):
    seed.seed_everything(1)
    wandb.login()
    run = wandb.init(
        project=hparams.project_name
    )   
    wandb_logger = WandbLogger()
    wandb.config.update(hparams)

    encoder = SentenceTransformer(hparams.sbert)

    with open(f'../{hparams.emb_data}', "rb") as fIn:
        stored_data = pickle.load(fIn)
        datasets = {
            'train1': stored_data['train1'],
            'train2': stored_data['train2'],
            'val1': stored_data['val1'],
            'val2': stored_data['val2'],
        }

    dm = MyDataModule(datasets, encoder, hparams.batch_size, denoising=hparams.denoising)
    # model = AutoEncoder(768, hparams.hidden_dim, hparams.dropout, hparams.lr)
    model = AutoEncoder(768, hparams.hidden_dim, hparams.lr)
    filename = f'ae-quora-den-{wandb.run.name}'
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(
        dirpath='ae_saved/',
        monitor='val_loss',
        mode='min',
        # filename='test')
        filename=filename
    )
    trainer = pl.Trainer(
        logger=wandb_logger,
        gpus=1,
        max_epochs=hparams.epochs,
        deterministic=True,
        callbacks=[checkpoint_callback,
            # EarlyStopping(monitor='val_loss', patience=hparams.patience, mode='min'),
            lr_monitor,
        ],
        # check_val_every_n_epoch=5,
        auto_lr_find=True,
    )
    # trainer.tune(model,dm)
    trainer.fit(model, dm)
    # trainer.test(datamodule=dm)
    print(checkpoint_callback.best_model_score)
    wandb.log({'best_loss': checkpoint_callback.best_model_score})

    run.finish()

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--sbert', type=str, default='stsb-xlm-r-multilingual') #paraphrase-xlm-r-multilingual-v1 / stsb-xlm-r-multilingual
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=8e-5)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--denoising', type=bool, default=False)
    parser.add_argument('--emb_data', type=str, default='quora.pkl')
    parser.add_argument('--project_name', type=str, default='ae_ae')
    parser.add_argument('--dropout', type=float, default=0.2)
    args = parser.parse_args()

    main(args)