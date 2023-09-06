import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import ViTModel, ViTConfig,\
    DeiTModel, DeiTConfig, SwinModel, SwinConfig

class ViT(nn.Module):

  def __init__(self, config=ViTConfig(), num_labels=3, 
               model_checkpoint='google/vit-base-patch16-224-in21k'):

        super(ViT, self).__init__()

        self.vit = ViTModel.from_pretrained(model_checkpoint, add_pooling_layer=False)
        self.classifier = (
            nn.Linear(config.hidden_size, num_labels) 
        )

  def forward(self, x):

    x = self.vit(x)['last_hidden_state']
    # Use the embedding of [CLS] token
    output = self.classifier(x[:, 0, :])

    return output
  

class DEiT(nn.Module):

  def __init__(self, config=DeiTConfig(), num_labels=3, 
               model_checkpoint='facebook/deit-base-distilled-patch16-224'):

        super(DEiT, self).__init__()

        self.deit = DeiTModel.from_pretrained(model_checkpoint, add_pooling_layer=False)
        self.classifier = (
            nn.Linear(config.hidden_size, num_labels) 
        )

  def forward(self, x):

    x = self.deit(x)['last_hidden_state']
    # Use the embedding of [CLS] token
    output = self.classifier(x[:, 0, :])

    return output
  
  
class SwinT(nn.Module):

  def __init__(self, config=SwinConfig(), num_labels=3, 
               model_checkpoint='microsoft/swin-tiny-patch4-window7-224'):

        super(SwinT, self).__init__()

        self.swin = SwinModel.from_pretrained(model_checkpoint, add_pooling_layer=False)
        self.classifier = (
            nn.Linear(config.hidden_size, num_labels) 
        )

  def forward(self, x):

    x = self.swin(x)['last_hidden_state']
    # Use the embedding of [CLS] token
    output = self.classifier(x[:, 0, :])

    return output