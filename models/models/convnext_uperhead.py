import torch
from torch import Tensor
from torch.nn import functional as F
from semseg.models.base import BaseModel
from semseg.models.heads import UPerHead


class CustomCNN(BaseModel):
    def __init__(self, backbone: str = 'ResNet-50', num_classes: int = 19):
        super().__init__(backbone, num_classes)
        self.backbone.load_state_dict(torch.load("models/models/convnext_base_1k_224_ema.pth")["model"], strict=False)
        self.decode_head = UPerHead(self.backbone.channels, 256, num_classes)
        self.apply(self._init_weights)

    def forward(self, x: Tensor) -> Tensor:
        y = self.backbone(x)
        y = self.decode_head(y)   # 4x reduction in image size
        y = F.interpolate(y, size=x.shape[2:], mode='bilinear', align_corners=False)    # to original image shape
        return y

    
def convnext_uperhead(args):
    model = CustomCNN('ConvNeXt-B', args.num_classes)
    #model.init_pretrained("models/models/convnext_base_1k_224_ema.pth")
    return model