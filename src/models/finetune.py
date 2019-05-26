import torch
import torch.nn as nn
from cnn_finetune import make_model


class Finetune(nn.Module):
    def __init__(self, model_name, num_classes=5):
        super(Finetune, self).__init__()
        self.model = make_model(
            model_name=model_name,
            input_size=(224, 224),
            pretrained=True,
            num_classes=num_classes
        )

    def freeze(self):
        for param in self.model._features.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.model._features.parameters():
            param.requires_grad = True

    def forward(self, x):
        return self.model(x)