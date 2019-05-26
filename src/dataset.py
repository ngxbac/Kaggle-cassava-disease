import numpy as np
import os
import cv2
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder


def load_image(path):
    try:
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except:
        print(path)
        image = np.zeros((256, 256, 3)).astype(np.float32)
    return image


class CassavaDataset(Dataset):
    def __init__(self, df, root, transform, mode, stage=1):
        self.df = pd.read_csv(df, nrows=None)
        self.root = root
        self.transform = transform
        self.mode = mode

        le = LabelEncoder()
        self.df['class'] = le.fit_transform(self.df['class'])

        self.ids = self.df['files'].values
        if self.mode == 'train' or self.mode == 'valid':
            self.labels = self.df['class'].values
        else:
            self.labels = [0] * len(self.ids)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        id = self.ids[idx]
        label = self.labels[idx]

        image = load_image(os.path.join(self.root, id))
        if self.transform:
            image = self.transform(image=image)['image']
            image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        return {
            "images": image,
            "targets": label
        }
