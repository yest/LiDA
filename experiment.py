import os

ae_model = 'restful-rain-90'  # Pretrained autoencoder model
da_model = 'brisk-bush-1'  # Pretrained denoising autoencoder model
aug_number = 0.2 # Small number for linear transformation
ae_hidden = 768
dropout = 0.5
epochs = 100 # number of epochs
samples = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] # datasets fractions
langs = ['en', 'cn', 'id'] # dataset language, en: English, cn: Chinese, id: Indonesian
project_name = 'all_languages_new' # project name for wandb
aug_types = ['all'] # types of data augmentation techniques, 'linear', 'ae', 'da', 'all'

for dataset in langs:
    for sample in samples:
        lr = 9e-6
        for aug_type in aug_types:
            os.system(
                f'python main.py --augmenting=True --sample={sample} --dataset={dataset} --lr={lr} --ae_model={ae_model} --ae_hidden={ae_hidden} --dropout={dropout} --project_name={project_name} --epochs={epochs} --da_model={da_model} --aug_number={aug_number} --aug_type={aug_type}')
