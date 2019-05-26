import pandas as pd
import numpy as np

import torch
import torch.nn.functional as Ftorch
from torch.utils.data import DataLoader
import os
import glob
import click
from tqdm import *

from models import Finetune
from augmentation import *
from dataset import CassavaDataset


device = torch.device('cuda')


@click.group()
def cli():
    print("Making submission")


labels = sorted(['cmd', 'healthy', 'cgm', 'cbsd', 'cbb'])
i2c = {}
for i, label in enumerate(labels):
    i2c[i] = label


def predict(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for dct in tqdm(loader, total=len(loader)):
            images = dct['images'].to(device)
            pred = model(images)
            pred = Ftorch.softmax(pred)
            pred = pred.detach().cpu().numpy()
            preds.append(pred)

    preds = np.concatenate(preds, axis=0)
    return preds


@cli.command()
def predict():
    test_csv = '/raid/bac/kaggle/cassava-disease/notebooks/csv/test.csv'
    log_dir = "/raid/bac/kaggle/logs/cassava-disease/finetune/"

    all_preds = []
    for model_name in ['resnet50', 'se_resnet50', 'densenet121']:

        test_augs = [valid_aug(320)]

        one_model_kfold = []
        for fold in range(5):
            model = Finetune(
                model_name=model_name,
                num_classes=5,
            )

            all_checkpoints = glob.glob(f"{log_dir}/{model_name}/fold_{fold}/checkpoints/stage2*")
            for checkpoint in all_checkpoints:
                checkpoint = torch.load(checkpoint)
                model.load_state_dict(checkpoint['model_state_dict'])
                model = model.to(device)

                for tta in test_augs:
                    # Dataset
                    dataset = CassavaDataset(
                        df=test_csv,
                        root='/',
                        transform=tta,
                        mode='infer'
                    )

                    loader = DataLoader(
                        dataset=dataset,
                        batch_size=128,
                        shuffle=False,
                        num_workers=4,
                    )

                    fold_pred = predict(model, loader)
                    one_model_kfold.append(fold_pred)

        one_model_kfold = np.stack(one_model_kfold, axis=0).mean(axis=0)
        np.save(f"./submission/{model_name}.csv", one_model_kfold)
        all_preds.append(one_model_kfold)
    all_preds = np.stack(all_preds, axis=0).mean(axis=0)

    all_preds = np.argmax(all_preds, axis=1)
    all_preds = [i2c[i] for i in all_preds]
    submission = dataset.df.copy()
    submission['Id'] = submission['files'].apply(lambda x: x.split("/")[-1])
    submission['Category'] = all_preds
    os.makedirs('submission', exist_ok=True)
    submission[['Id', 'Category']].to_csv(f'./submission/ensemble_kfold.csv', index=False)


@cli.command()
def from_numpy():
    test_csv = '/raid/bac/kaggle/cassava-disease/notebooks/csv/test.csv'
    submission = pd.read_csv(test_csv)
    all_preds = []
    for model_name in ['resnet50', 'se_resnet50', 'densenet121']:
        one_model_kfold = np.load(f"./submission/{model_name}.csv.npy")
        all_preds.append(one_model_kfold)
    all_preds = np.stack(all_preds, axis=0).mean(axis=0)

    all_preds = np.argmax(all_preds, axis=1)
    all_preds = [i2c[i] for i in all_preds]
    submission['Id'] = submission['files'].apply(lambda x: x.split("/")[-1])
    submission['Category'] = all_preds
    os.makedirs('submission', exist_ok=True)
    submission[['Id', 'Category']].to_csv(f'./submission/ensemble_kfold.csv', index=False)


if __name__ == '__main__':
    cli()