import torch
from .models.simple_unet import UNET
from .models.ddrnet import DDRNet
from .models.network.ocrnet import HRNet_Mscale
#from .models.network.ocrnet import HRNet_Mscale
import segmentation_models_pytorch as smp
from torchsummary import summary

def HRNet_Mscale(num_classes, criterion):
    return MscaleOCR(num_classes, trunk='hrnetv2', criterion=criterion)

ENCODER = "efficientnet-b0"
ENCODER_WEIGHTS = "imagenet"

def get_model(args):
    if args.model == "unet":
        model = smp.Unet(
            encoder_name=ENCODER,        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights=ENCODER_WEIGHTS,     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=args.num_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=args.num_classes,                      # model output channels (number of classes in your dataset)
        )
    elif args.model == "deeplab":
        model = smp.DeepLabV3Plus(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=args.num_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=args.num_classes,
        )
    elif args.model == "unetplusplus":
        model = smp.UnetPlusPlus(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=args.num_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=args.num_classes,
        )
    elif args.model == "manet":
        model = smp.MAnet(
            encoder_name=ENCODER,
            encoder_weights=ENCODER_WEIGHTS,
            in_channels=args.num_channels,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=args.num_classes,
        )
    elif args.model == "ddrnet":
        return DDRNet(num_classes=args.num_classes)
    elif args.model == "mscale":
        return HRNet_Mscale(num_classes=args.num_classes, criterion=None)
        
    #elif model == "mscale":
    #    return HRNet_Mscale(2)
    
    print(summary(model.cuda(), (args.num_channels, args.image_dim, args.image_dim)))
        
    return model
    