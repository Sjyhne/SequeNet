import torch
from .models.simple_unet import UNET

def get_model(model):
    if model == "deeplab":
        model = torch.hub.load('pytorch/vision:v0.8.0', 'deeplabv3_resnet50', pretrained=True)
        return model
    elif model == "unet":
        return UNET