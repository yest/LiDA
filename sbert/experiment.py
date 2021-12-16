import os

ae_model = 'restful-rain-90'  # Pretrained autoencoder model
da_model = 'brisk-bush-1'  # Pretrained denoising autoencoder model
aug_number = 0.15 # Small number for linear transformation
ae_hidden = 768
dropout = 0.5
epochs = 10 # number of epochs
samples = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # datasets fractions
langs = ['en', 'cn', 'id'] # dataset language, en: English, cn: Chinese, id: Indonesian
project_name = 'sbert' # project name for wandb

for dataset in langs:
    for sample in samples:
        lr = 1e-6
        os.system(f'python main.py --augmenting=True --sample={sample} --dataset={dataset} --lr={lr} --ae_model={ae_model} --ae_hidden={ae_hidden} --project_name={project_name} --epochs={epochs}  --da_model={da_model} --aug_number={aug_number}')