import os

ae_model = 'restful-rain-90'  # AE
da_model = 'brisk-bush-1'  # Denoising
# da_model = 'spring-night-6'
aug_number = 0.2


ae_hidden = 768
dropout = 0.5

dataset = 'id'
project_name = f'{dataset}-all'
epochs = 100
# samples = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
samples = [0.5]
# aug_type = 'all'


# for all language
# langs = ['en', 'cn', 'id']
langs = ['en']
project_name = 'all_languages_new'
# project_name = 'backtrans'
# project_name = 'eda'
# aug_types = ['linear', 'ae', 'da']
aug_types = ['all']
backtrans = False
eda = False
for dataset in langs:
    for sample in samples:
        # lr = 1e-5
        # os.system(f'python main_new.py --sample={sample} --dataset={dataset} --lr={lr} --dropout={dropout} --project_name={project_name} --epochs={epochs}')
        
        lr = 9e-6
        for aug_type in aug_types:
            os.system(
                f'python main.py --augmenting=True --sample={sample} --dataset={dataset} --lr={lr} --ae_model={ae_model} --ae_hidden={ae_hidden} --dropout={dropout} --project_name={project_name} --epochs={epochs} --da_model={da_model} --aug_number={aug_number} --aug_type={aug_type}')

        # backtrans
        # lr = 1e-5
        # os.system(f'python main_new.py --sample={sample} --dataset={dataset} --lr={lr} --dropout={dropout} --project_name={project_name} --epochs={epochs} --backtrans={backtrans}')


        # eda
        # lr = 8e-6
        # os.system(f'python main_new.py --sample={sample} --dataset={dataset} --lr={lr} --dropout={dropout} --project_name={project_name} --epochs={epochs} --eda_aug={eda}')
    
    

    # linear
    # lr = 1e-5
    # aug_number = 0.1
    # aug_type = 'linear'
    # epochs = 100
    # for i in range(20):
    #     os.system(
    #     f'python main_new.py --augmenting=True --sample={sample} --dataset={dataset} --lr={lr} --dropout={dropout} --project_name={project_name} --epochs={epochs} --aug_number={aug_number} --aug_type={aug_type}')
    #     aug_number  +=  0.01
