import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
import os
import glob
import click
from tqdm import *

from models import Finetune
from augmentation import *
from dataset import CassavaDataset


device = torch.device('cuda')


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
            pred = pred.detach().cpu().numpy()
            preds.append(pred)

    preds = np.concatenate(preds, axis=0)
    return preds


if __name__ == '__main__':

    test_csv = '/raid/bac/kaggle/cassava-disease/notebooks/csv/test.csv'
    log_dir = "/raid/bac/kaggle/logs/cassava-disease/finetune/"
    model_name = 'resnet50'

    test_augs = test_tta(320)

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
    one_model_pred = np.argmax(one_model_kfold, axis=1)
    one_model_pred_cls = [i2c[i] for i in one_model_pred]
    submission = dataset.df.copy()
    submission['Id'] = submission['files'].apply(lambda x: x.split("/")[-1])
    submission['Category'] = one_model_pred_cls
    os.makedirs('submission', exist_ok=True)
    submission[['Id', 'Category']].to_csv(f'./submission/{model_name}_kfold.csv', index=False)



