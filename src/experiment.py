import random
import torch
import torch.nn as nn
from collections import OrderedDict
from catalyst.dl.experiments import ConfigExperiment
from src.augmentation import *
from src.dataset import *


class Experiment(ConfigExperiment):

    def _postprocess_model_for_stage(self, stage: str, model: nn.Module):

        import warnings
        warnings.filterwarnings('ignore')

        random.seed(2411)
        np.random.seed(2411)
        torch.manual_seed(2411)

        model_ = model
        if isinstance(model, torch.nn.DataParallel):
            model_ = model_.module

        if stage in ["debug", "stage1"]:
            model_.freeze()
        elif stage == "stage2":
            model_.unfreeze()

        return model_

    def get_datasets(self, stage: str, **kwargs):
        datasets = OrderedDict()

        train_csv = kwargs.get('train_csv', None)
        valid_csv = kwargs.get('valid_csv', None)
        datapath = kwargs.get('datapath', None)

        trainset = CassavaDataset(
            df=train_csv,
            root=datapath,
            mode='train',
            transform=train_aug(320),
        )
        testset = CassavaDataset(
            df=valid_csv,
            root=datapath,
            mode='valid',
            transform=valid_aug(320),
        )

        datasets["train"] = trainset
        datasets["valid"] = testset

        return datasets
