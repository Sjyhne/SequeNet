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

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    jaccard = tf.concat((jaccard[0:1], jaccard[1:] - jaccard[:-1]), 0)
    return jaccard

def lovasz_softmax(labels, probas, classes='present', per_image=False, ignore=None, order='BCHW'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, H, W, C] or [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
      order: use BHWC or BCHW
    """
    if per_image:
        def treat_image(prob_lab):
            prob, lab = prob_lab
            prob, lab = tf.expand_dims(prob, 0), tf.expand_dims(lab, 0)
            prob, lab = flatten_probas(prob, lab, ignore, order)
            return lovasz_softmax_flat(prob, lab, classes=classes)
        losses = tf.map_fn(treat_image, (probas, labels), dtype=tf.float32)
        loss = tf.reduce_mean(losses)
    else:
        loss = lovasz_softmax_flat(probas, labels, classes=classes)
    return loss


def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """
    C = probas.shape[1]
    losses = []
    present = []
    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = tf.cast(tf.equal(labels, c), probas.dtype)  # foreground for class c
        if classes == 'present':
            present.append(tf.reduce_sum(fg) > 0)
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]
        errors = tf.abs(fg - class_pred)
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], name="descending_sort_{}".format(c))
        fg_sorted = tf.gather(fg, perm)
        grad = lovasz_grad(fg_sorted)
        losses.append(
            tf.tensordot(errors_sorted, tf.stop_gradient(grad), 1, name="loss_class_{}".format(c))
                      )
    if len(class_to_sum) == 1:  # short-circuit mean when only one class
        return losses[0]
    losses_tensor = tf.stack(losses)
    if classes == 'present':
        present = tf.stack(present)
        losses_tensor = tf.boolean_mask(losses_tensor, present)
    loss = tf.reduce_mean(losses_tensor)
    return loss


def flatten_probas(probas, labels, ignore=None, order='BHWC'):
    """
    Flattens predictions in the batch
    """
    if len(probas.shape) == 3:
        probas, order = tf.expand_dims(probas, 3), 'BHWC'
    if order == 'BCHW':
        print(probas.shape)
        probas = tf.transpose(probas, (0, 2, 3, 1), name="BCHW_to_BHWC")
        order = 'BHWC'
    if order != 'BHWC':
        raise NotImplementedError('Order {} unknown'.format(order))
    C = probas.shape[3]
    probas = tf.reshape(probas, (-1, C))
    labels = tf.reshape(labels, (-1,))
    if ignore is None:
        return probas, labels
    valid = tf.not_equal(labels, ignore)
    vprobas = tf.boolean_mask(probas, valid, name='valid_probas')
    vlabels = tf.boolean_mask(labels, valid, name='valid_labels')
    return vprobas, vlabels


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


from keras import backend as K
import numpy as np
import tensorflow as tf
from scipy.ndimage import distance_transform_edt as distance


def calc_dist_map(seg):
    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

    return res


def calc_dist_map_batch(y_true):
    y_true_numpy = y_true.numpy()
    return np.array([calc_dist_map(y)
                     for y in y_true_numpy]).astype(np.float32)


def surface_loss_keras(y_true, y_pred, distmap):
    #y_true_dist_map = tf.py_function(func=calc_dist_map_batch,
    #                                 inp=[y_true],
    #                                 Tout=tf.float32)
    multipled = y_pred * distmap
    return K.mean(multipled)

class TFABL(tf.keras.losses.Loss):
    def __init__(self):
        ...
    
    def call(self, target, logits):
        return surface_loss_keras(target, logits)


# Active Boundary Loss
class ABL(tf.keras.losses.Loss):
    def __init__(self, isdetach=True, max_N_ratio = 1/100, ignore_label = 255, label_smoothing=0, weight = None, max_clip_dist = 20.):
        super(ABL, self).__init__()
        self.ignore_label = ignore_label
        self.label_smoothing = label_smoothing
        self.isdetach=isdetach
        self.max_N_ratio = max_N_ratio

        self.weight_func = lambda w, max_distance=max_clip_dist: tf.clip_by_value(w, clip_value_min=-1000000, clip_value_max=max_distance) / max_distance

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
            self.criterion = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True,
                reduction=tf.keras.losses.Reduction.NONE,

            )
        else:
            self.criterion = lovasz_softmax
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
        # dilate
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

        zero = tf.constant(0, dtype=tf.int16)
        where = tf.not_equal(pred_bound*1, zero)
        bound = tf.where(where)

        n,x,y = tf.transpose(bound)

        max_dis = 1e5
        logits = tf.transpose(logits, perm=[0,2,3,1]) # NHWC

        pred_dist_map_d = tf.pad(pred_dist_map,[[0,0],[1,1],[1,1]],mode='CONSTANT', constant_values=max_dis) # NH+2W+2

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

                logits_now = tf.gather_nd(indices=indice_pairs, params=logits_d)
                kl_map_now = kl_div(kl_center, logits_now)
                
                kl_map_now = tf.math.reduce_sum(kl_map_now, axis=1)  # KC->K

                kl_maps = tf.concat((kl_maps, tf.expand_dims(kl_map_now, axis=0)), 0)
                tf.clip_by_value(kl_maps, clip_value_min=0.0, clip_value_max=20.0)

        # direction_gt shound be Nk  (8k->K)
        direction_gt = tf.math.argmin(dist_maps, axis=0)
        weight_ce = tf.gather_nd(indices=center_indice_pairs, params=pred_dist_map)

        # delete if min is 8 (local position)
        direction_gt_idx = [direction_gt!=8][0]
        direction_gt = tf.boolean_mask(direction_gt, direction_gt_idx)
        kl_maps = tf.transpose(kl_maps, perm=[1, 0])
        direction_pred = kl_maps[direction_gt_idx]
        weight_ce = weight_ce[direction_gt_idx]
        return direction_gt, direction_pred, weight_ce

    def get_dist_maps(self, target):
        target_detach = target
        dist_maps = tf.concat([dist_map_transform(target_detach[i]) for i in range(target_detach.shape[0])], axis=0)
        out = -dist_maps
        out = tf.where(out>0, out, tf.zeros_like(out))
        
        return out

    def call(self, target, logits):
        logits = tf.transpose(logits, perm=[0, 3, 1, 2])

        if len(target.shape) > 3:
            target = tf.math.argmax(target, axis=-1)
        
        gt_boundary = self.gt2boundary(target, ignore_label=self.ignore_label)

        dist_maps = self.get_dist_maps(gt_boundary)

        pred_boundary = tf.cast(self.logits2boundary(logits), dtype=tf.int16)
        if tf.math.reduce_sum(pred_boundary) < 1: # avoid nan
            return None # you should check in the outside. if None, skip this loss.

        direction_gt, direction_pred, weight_ce = self.get_direction_gt_predkl(dist_maps, pred_boundary, logits) # NHW,NHW,NCHW

        loss = self.criterion(direction_gt, direction_pred) # careful

        weight_ce = self.weight_func(weight_ce)

        loss = tf.math.reduce_mean(loss * weight_ce)  # add distance weight

        return loss


if __name__ == '__main__':
    n,h,w,c = 8,512,512,2
    gt = np.zeros((n,h,w,c))
    gt[0,5:10] = 1
    gt[0,50:60] = 1
    gt = tf.convert_to_tensor(gt, dtype=tf.float32)
    logits = np.zeros((n, h, w, c))
    logits[0, 5:30, 20, 0] = 3.41
    logits[0, 5:30, 20, 1] = 51.2
    logits[0, 55, 30, 0] = 54.2
    logits[0, 55, 30, 1] = 0.43
    logits = tf.convert_to_tensor(logits, dtype=tf.float32)


    #f, x = plt.subplots(1, 2)
    #x[0].imshow(gt[0])
    #x[1].imshow(tf.transpose(logits[0]))

    #plt.show()


    abl = ABL(label_smoothing=0)
    print(abl(gt, logits))
