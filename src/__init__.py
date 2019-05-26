
# flake8: noqa
from catalyst.dl import registry

from .experiment import Experiment
from .runner import ModelRunner as Runner
from .callbacks import *
from .models import *
from .losses import *

registry.Model(Net)
registry.Model(FewShotModel)
registry.Model(Finetune)
registry.Callback(MixupLossCallback)
registry.Callback(IterCheckpointCallback)
registry.Criterion(FocalLoss)