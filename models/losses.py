from ast import operator
import tensorflow as tf

import numpy as np
from scipy.ndimage import distance_transform_edt as distance
# can find here: https://github.com/CoinCheung/pytorch-loss/blob/af876e43218694dc8599cc4711d9a5c5e043b1b2/label_smooth.py
#from .label_smooth import LabelSmoothSoftmaxCEV1 as LSSCE
#from torchvision import transforms
from functools import partial
from operator import itemgetter

import matplotlib.pyplot as plt

# Tools
def kl_div(a,b): # q,p
    return tf.nn.softmax(b, axis=1) * (tf.nn.log_softmax(b, axis=1) - tf.nn.log_softmax(a, axis=1))   

def one_hot2dist(seg):
    res = np.zeros_like(seg)
    for i in range(len(seg)):
        posmask = seg[i].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            res[i] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res

def class2one_hot(seg, C):
    seg = tf.expand_dims(seg, axis=0) if len(seg.shape) == 2 else seg
    res = tf.cast(tf.stack([seg == c for c in range(C)], axis=1), tf.int32)
    return res

def dist_map_transform(value):
    value = tf.expand_dims(value, axis=0)
    value = tf.cast(value, dtype=tf.int64)
    value = class2one_hot(value, C=1)
    getter = itemgetter(0)
    value = getter(value)
    #value = value.cpu().numpy()
    value = value.numpy()
    value = one_hot2dist(value)
    return tf.convert_to_tensor(value, dtype=tf.float32)

# Active Boundary Loss
class ABL(tf.keras.losses.Loss):
    def __init__(self, isdetach=True, max_N_ratio = 1/100, ignore_label = 255, label_smoothing=0.2, weight = None, max_clip_dist = 20.):
        super(ABL, self).__init__()
        self.ignore_label = ignore_label
        self.label_smoothing = label_smoothing
        self.isdetach=isdetach
        self.max_N_ratio = max_N_ratio

        self.weight_func = lambda w, max_distance=max_clip_dist: tf.clip_by_value(w, clip_value_min=-100, clip_value_max=max_distance) / max_distance

        """
        self.dist_map_transform = transforms.Compose([
            lambda img: img.unsqueeze(0),
            lambda nd: nd.type(tf.int64),
            partial(class2one_hot, C=1),
            itemgetter(0),
            lambda t: t.cpu().numpy(),
            one_hot2dist,
            lambda nd: tf.Tensor(nd, dtype=tf.float32)
        ])
        """

        if label_smoothing == 0:
            self.criterion = tf.keras.losses.CategoricalCrossentropy(
                reduction=tf.keras.losses.Reduction.NONE
            )
        else:
            self.criterion = tf.keras.losses.CategoricalCrossentropy(
                reduction=tf.keras.losses.Reduction.NONE,
                label_smoothing=label_smoothing
            )
            #self.criterion = LSSCE(
            #    reduction='none',
            #    lb_smooth = label_smoothing
            #)

    def logits2boundary(self, logit):
        eps = 1e-5
        _, _, h, w = logit.shape
        max_N = (h*w) * self.max_N_ratio
        kl_lr = tf.math.reduce_sum(kl_div(logit[:, :, 1:, :], logit[:, :, :-1, :]), axis=1, keepdims=True)
        kl_ud = tf.math.reduce_sum(kl_div(logit[:, :, :, 1:], logit[:, :, :, :-1]), axis=1, keepdims=True)
        kl_lr = tf.pad(
            kl_lr, [[0, 0], [0, 0], [0, 1], [0, 0]], mode='CONSTANT', constant_values=0)
        kl_ud = tf.pad(
            kl_ud, [[0, 0], [0, 0], [0, 0], [0, 1]], mode='CONSTANT', constant_values=0)
        kl_combine = kl_lr+kl_ud
        while True: # avoid the case that full image is the same color
            kl_combine_bin = tf.cast((kl_combine > eps), dtype=tf.float32)
            if tf.math.reduce_sum(kl_combine_bin) > max_N:
                eps *=1.2
            else:
                break
        #dilate
        #dilate_weight = tf.ones((1,1,3,3)).cuda()
        dilate_weight = tf.ones((3,3,1,1))
        kl_combine_bin = tf.transpose(kl_combine_bin, perm=[0, 2, 3, 1])
        edge2 = tf.nn.conv2d(kl_combine_bin, dilate_weight, strides=1, data_format="NHWC", padding="SAME")
        edge2 = tf.transpose(edge2, perm=[0, 3, 1, 2])
        edge2 = tf.squeeze(edge2, axis=1)  # NCHW->NHW
        kl_combine_bin = (edge2 > 0)
        return kl_combine_bin

    def gt2boundary(self, gt, ignore_label=-1):  # gt NHW
        gt_lr = gt[:,1:,:]-gt[:,:-1,:]  # NHW
        gt_ud = gt[:,:,1:]-gt[:,:,:-1]

        gt_lr = tf.cast(gt_lr, dtype=tf.int16)
        gt_ud = tf.cast(gt_ud, dtype=tf.int16)
        gt_lr = tf.pad(gt_lr, [[0,0],[0,1],[0,0]], mode='CONSTANT', constant_values=0) != 0 
        gt_ud = tf.pad(gt_ud, [[0,0],[0,0],[0,1]], mode='CONSTANT', constant_values=0) != 0
        gt_lr = tf.cast(gt_lr, dtype=tf.int16)
        gt_ud = tf.cast(gt_ud, dtype=tf.int16)

        gt_combine = gt_lr+gt_ud
        del gt_lr
        del gt_ud
        
        gt = tf.cast(gt, dtype=tf.int16)

        # set 'ignore area' to all boundary
        gt_combine += tf.cast((gt==ignore_label), dtype=tf.int16)
        
        return gt_combine > 0

    def get_direction_gt_predkl(self, pred_dist_map, pred_bound, logits):
        # NHW,NHW,NCHW
        eps = 1e-5
        # bound = torch.where(pred_bound)  # 3k
        #bound = torch.nonzero(pred_bound*1) # TODO: FIX
        
        zero = tf.constant(0, dtype=tf.int16)
        where = tf.not_equal(pred_bound*1, zero)
        bound = tf.where(where)

        n,x,y = tf.transpose(bound)

        max_dis = 1e5
        logits = tf.transpose(logits, perm=[0,2,3,1]) # NHWC

        pred_dist_map_d = tf.pad(pred_dist_map,[[0,0],[1,1],[1,1]],mode='CONSTANT', constant_values=max_dis) # NH+2W+2

        #logits_d = tf.pad(logits,[[0,0],[1,1],[1,1],[0,0]],mode='CONSTANT') # N(H+2)(W+2)C
        logits_d = logits
        logits_d = tf.concat([logits_d, tf.expand_dims(logits_d[:,1,:,:], axis=1)], axis=1) # N(H+2)(W+2)C
        logits_d = tf.concat([tf.expand_dims(logits_d[:,-1,:,:], axis=1), logits_d], axis=1) # N(H+2)(W+2)C
        
        logits_d = tf.transpose(logits_d, perm=[0, 2, 1, 3])
        
        logits_d = tf.concat([logits_d, tf.expand_dims(logits_d[:,1,:,:], axis=1)], axis=1) # N(H+2)(W+2)C
        logits_d = tf.concat([tf.expand_dims(logits_d[:,-1,:,:], axis=1), logits_d], axis=1) # N(H+2)(W+2)C
        
        logits_d = tf.transpose(logits_d, perm=[0, 2, 1, 3])
        
        """
        | 4| 0| 5|
        | 2| 8| 3|
        | 6| 1| 7|
        """
        x_range = [1, -1,  0, 0, -1,  1, -1,  1, 0]
        y_range = [0,  0, -1, 1,  1,  1, -1, -1, 0]

        dist_maps = tf.zeros((0,len(x)))
        kl_maps = tf.zeros((0,len(x)))

        center_indice_pairs = tf.convert_to_tensor([list(a) for a in zip(n.numpy(), x.numpy(), y.numpy())])

        kl_center = tf.gather_nd(indices=center_indice_pairs, params=logits)

        for i, (dx, dy) in enumerate(zip(x_range, y_range)):
            indice_pairs = tf.convert_to_tensor([list(a) for a in zip(n.numpy(), x.numpy() + dx + 1, y.numpy() + dy + 1)])
            dist_now = tf.gather_nd(indices=indice_pairs, params=pred_dist_map_d)
            dist_maps = tf.concat((dist_maps, tf.expand_dims(dist_now, axis=0)), 0)

            if dx != 0 or dy != 0:

                #logits_now = logits_d[(n,x+dx+1,y+dy+1)]
                logits_now = tf.gather_nd(indices=indice_pairs, params=logits_d)
                # kl_map_now = torch.kl_div((kl_center+eps).log(), logits_now+eps).sum(2)  # 8KC->8K
                #if self.isdetach:
                #    logits_now = logits_now.detach()
                kl_map_now = kl_div(kl_center, logits_now)
                
                kl_map_now = tf.math.reduce_sum(kl_map_now, axis=1)  # KC->K
                kl_maps = tf.concat((kl_maps, tf.expand_dims(kl_map_now, axis=0)), 0)
                tf.clip_by_value(kl_maps, clip_value_min=0.0, clip_value_max=20.0)

        # direction_gt shound be Nk  (8k->K)
        direction_gt = tf.math.argmin(dist_maps, axis=0)
        # weight_ce = pred_dist_map[bound]
        #weight_ce = pred_dist_map[(n,x,y)]
        weight_ce = tf.gather_nd(indices=center_indice_pairs, params=pred_dist_map)

        # delete if min is 8 (local position)
        direction_gt_idx = [direction_gt!=8][0]
        direction_gt = tf.boolean_mask(direction_gt, direction_gt_idx)
        kl_maps = tf.transpose(kl_maps, perm=[1, 0])
        direction_pred = kl_maps[direction_gt_idx]
        weight_ce = weight_ce[direction_gt_idx]
        return direction_gt, direction_pred, weight_ce

    def get_dist_maps(self, target):
        #target_detach = target.clone().detach()
        target_detach = target
        dist_maps = tf.concat([dist_map_transform(target_detach[i]) for i in range(target_detach.shape[0])], axis=0)
        out = -dist_maps
        out = tf.where(out>0, out, tf.zeros_like(out))
        
        return out

    def call(self, target, logits):
        eps = 1e-10
        #ph, pw = logits.size(2), logits.size(3)
        #h, w = target.size(1), target.size(2)

        # Predicted height will be same as height
        #if ph != h or pw != w:
        #    logits = F.interpolate(input=logits, size=(
        #        h, w), mode='bilinear', align_corners=True)

        logits = tf.transpose(logits, perm=[0, 3, 1, 2])
        
        if len(target.shape) == 4:
            target = tf.squeeze(target, axis=-1)

        gt_boundary = self.gt2boundary(target, ignore_label=self.ignore_label)

        #dist_maps = self.get_dist_maps(gt_boundary).cuda() # <-- it will slow down the training, you can put it to dataloader.
        dist_maps = self.get_dist_maps(gt_boundary)
        pred_boundary = tf.cast(self.logits2boundary(logits), dtype=tf.int16)
        if tf.math.reduce_sum(pred_boundary) < 1: # avoid nan
            return None # you should check in the outside. if None, skip this loss.
        
        direction_gt, direction_pred, weight_ce = self.get_direction_gt_predkl(dist_maps, pred_boundary, logits) # NHW,NHW,NCHW
        direction_gt = tf.one_hot(direction_gt, 8)

        # direction_gt [K, 8], direction_pred [K, 8]
        loss = self.criterion(direction_gt, direction_pred) # careful

        weight_ce = self.weight_func(weight_ce)
        loss = tf.math.reduce_mean(loss * weight_ce)  # add distance weight

        return loss


if __name__ == '__main__':
    n,h,w,c = 1,512,512,2
    gt = np.zeros((n,h,w))
    gt[0,5] = 1
    gt[0,50] = 1
    gt = tf.convert_to_tensor(gt)
    logits = tf.random.normal((n,h,w,c))

    #f, x = plt.subplots(1, 2)
    #x[0].imshow(gt[0])
    #x[1].imshow(tf.transpose(logits[0]))

    #plt.show()


    abl = ABL(label_smoothing=0.2)
    print(abl(gt, logits))
