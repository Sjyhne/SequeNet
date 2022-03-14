import torch
from .models.simple_unet import UNET
import segmentation_models_pytorch as smp
from torchsummary import summary

ENCODER = "efficientnet-b0"
ENCODER_WEIGHTS = "imagenet"

def get_model(model):
    if model == "unet":
        model = smp.Unet(
            encoder_name=ENCODER,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=ENCODER_WEIGHTS,     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=2,                      # model output channels (number of classes in your dataset)
        )
    elif model == "deeplab":
        model = smp.DeepLabV3Plus(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=3,
            classes=2,
        )
    elif model == "unetplusplus":
        model = smp.UnetPlusPlus(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=3,
            classes=2,
        )
    elif model == "manet":
        model = smp.MAnet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=3,
            classes=2,
        )
    
    print(summary(model.cuda(), (3, 224, 224)))
        
    return model
    