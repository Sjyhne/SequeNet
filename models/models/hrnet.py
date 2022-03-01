import tensorflow as tf
from .hrnet_layers import (
    BasicBlock,
    Branch,
    FinalLayer,
    FuseLayer1,
    FuseLayer2,
    FuseLayer3,
    StemNet,
    TransitionLayer1,
    TransitionLayer2,
    TransitionLayer3
)


class HRNet(tf.keras.models.Model):

    def __init__(self, height=512, width=512, channel=3, classes=2):

        super(HRNet, self).__init__()
        self.StemNet = StemNet((None, height, width, channel))
        self.TransitionLayer1 = TransitionLayer1()
        self.TransitionLayer2 = TransitionLayer2()
        self.TransitionLayer3 = TransitionLayer3()

        self.FuseLayer1 = FuseLayer1()
        self.FuseLayer2 = FuseLayer2()
        self.FuseLayer3 = FuseLayer3()

        self.FinalLayer = FinalLayer(classes)

        self.Branch1_0 = Branch(32)
        self.Branch1_1 = Branch(64)

        self.Branch2_0 = Branch(32)
        self.Branch2_1 = Branch(64)
        self.Branch2_2 = Branch(128)

        self.Branch3_0 = Branch(32)
        self.Branch3_1 = Branch(64)
        self.Branch3_2 = Branch(128)
        self.Branch3_3 = Branch(256)

    def call(self, x):
        
        x = tf.cast(x, dtype=tf.float32)
        
        x = self.StemNet(x)

        x = self.TransitionLayer1(x)
        x0 = self.Branch1_0(x[0])
        x1 = self.Branch1_1(x[1])
        x = self.FuseLayer1([x0, x1])

        x = self.TransitionLayer2(x)
        x0 = self.Branch2_0(x[0])
        x1 = self.Branch2_1(x[1])
        x2 = self.Branch2_2(x[2])
        x = self.FuseLayer2([x0, x1, x2])

        x = self.TransitionLayer3(x)
        x0 = self.Branch3_0(x[0])
        x1 = self.Branch3_1(x[1])
        x2 = self.Branch3_2(x[2])
        x3 = self.Branch3_3(x[3])

        x = self.FuseLayer3([x0, x1, x2, x3])

        out = self.FinalLayer(x)

        return out

def hrnet(n_classes, input_height=None, input_width=None):
    model = HRNet(classes=n_classes, height=input_height, width=input_width)
    
    #print(model.summary())
    
    return model

def deeplab(n_classes, input_height=None, input_width=None, filters=256, final_activation="softmax", **kwargs):
    model = DeepLabV3plus(n_classes, input_height, input_width, "resnet152v2", filters)
    
    model = model.build()
    
    print(input_height, input_width)

    model = model.model()

    print(model.summary())

    return model
