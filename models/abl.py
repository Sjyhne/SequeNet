import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.ndimage import distance_transform_edt as distance
# can find here: https://github.com/CoinCheung/pytorch-loss/blob/af876e43218694dc8599cc4711d9a5c5e043b1b2/label_smooth.py
from .label_smooth import LabelSmoothSoftmaxCEV1 as LSSCE
from torchvision import transforms
from functools import partial
from operator import itemgetter
import matplotlib.pyplot as plt

# Tools
def kl_div(a,b): # q,p
    return F.softmax(b, dim=1) * (F.log_softmax(b, dim=1) - F.log_softmax(a, dim=1))   

def one_hot2dist(seg):
    res = np.zeros_like(seg)
    for i in range(len(seg)):
        posmask = seg[i].astype(bool)
        if posmask.any():
            negmask = ~posmask
            res[i] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res

def class2one_hot(seg, C):
    seg = seg.unsqueeze(dim=0) if len(seg.shape) == 2 else seg
    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    return res

# Active Boundary Loss
class ABL(nn.Module):
    def __init__(self, isdetach=True, max_N_ratio = 1/100, ignore_label = 255, label_smoothing=0.0, weight = None, max_clip_dist = 20.):
        super(ABL, self).__init__()
        self.ignore_label = ignore_label
        self.label_smoothing = label_smoothing
        self.isdetach=isdetach
        self.max_N_ratio = max_N_ratio
        
        # TODO: Set max max_N value

        self.weight_func = lambda w, max_distance=max_clip_dist: torch.clamp(w, max=max_distance) / max_distance

        self.dist_map_transform = transforms.Compose([
            lambda img: img.unsqueeze(0),
            lambda nd: nd.type(torch.int64),
            partial(class2one_hot, C=1),
            itemgetter(0),
            lambda t: t.cpu().numpy(),
            one_hot2dist,
            lambda nd: torch.tensor(nd, dtype=torch.float32)
        ])

        if label_smoothing == 0:
            self.criterion = nn.CrossEntropyLoss(
                weight=weight,
                ignore_index=ignore_label,
                reduction='none'
            )
        else:
            self.criterion = LSSCE(
                reduction='none',
                ignore_index=ignore_label,
                lb_smooth = label_smoothing
            )
        print(label_smoothing)

    def logits2boundary(self, logit, gt_boundary):
        n, _, h, w = logit.shape
        eps = torch.full((n,), 1e-4)
        batch_max_N = torch.where(gt_boundary == 0, 1, 0).sum((1, 2)) / 3 * 0.9
        kl_lr = kl_div(logit[:, :, 1:, :], logit[:, :, :-1, :]).sum(1, keepdim=True)
        kl_ud = kl_div(logit[:, :, :, 1:], logit[:, :, :, :-1]).sum(1, keepdim=True)
        kl_lr = torch.nn.functional.pad(kl_lr, [0, 0, 0, 1, 0, 0, 0, 0], mode='constant', value=0)
        kl_ud = torch.nn.functional.pad(kl_ud, [0, 1, 0, 0, 0, 0, 0, 0], mode='constant', value=0)
        kl_combine = kl_lr+kl_ud
        kl_combine = torch.squeeze(kl_combine)
        kl_combine_bin = torch.full((n, h, w), 0.0).cuda()
        for idx, _ in enumerate(kl_combine):
            while True: # avoid the case that full image is the same colo
                kl_combine_bin[idx] = (kl_combine[idx] > eps[idx]).to(torch.float32)
                if kl_combine_bin[idx].sum() > batch_max_N[idx]:
                    eps[idx] = eps[idx] * 1.5
                else:
                    break
                    
        kl_combine_bin = torch.unsqueeze(kl_combine_bin, dim=1)

        dilate_weight = torch.ones((1,1,3,3)).cuda()
        edge2 = torch.nn.functional.conv2d(kl_combine_bin, dilate_weight, stride=1, padding=1)
        edge2 = edge2.squeeze(1)  # NCHW->NHW
        kl_combine_bin = (edge2 > 0)
        return kl_combine_bin

    def gt2boundary(self, gt, ignore_label=-1):  # gt NHW
        gt_lr = gt[:,1:,:]-gt[:,:-1,:]  # NHW
        gt_ud = gt[:,:,1:]-gt[:,:,:-1]
        gt_lr = torch.nn.functional.pad(gt_lr, [0,0,0,1,0,0], mode='constant', value=0) != 0 
        gt_ud = torch.nn.functional.pad(gt_ud, [0,1,0,0,0,0], mode='constant', value=0) != 0
        gt_combine = gt_lr+gt_ud
        del gt_lr
        del gt_ud
        
        # set 'ignore area' to all boundary
        gt_combine += (gt==ignore_label)
        
        return gt_combine > 0

    def get_direction_gt_predkl(self, pred_dist_map, pred_bound, logits):
        # NHW,NHW,NCHW
        eps = 1e-5
        # bound = torch.where(pred_bound)  # 3k
        bound = torch.nonzero(pred_bound*1)
        n,x,y = bound.T
        max_dis = 1e5

        logits = logits.permute(0,2,3,1) # NHWC
        
        pred_dist_map_d = torch.nn.functional.pad(pred_dist_map,(1,1,1,1,0,0),mode='constant', value=max_dis) # NH+2W+2

        logits_d = torch.nn.functional.pad(logits,(0,0,1,1,1,1,0,0),mode='constant') # N(H+2)(W+2)C
        logits_d[:,0,:,:] = logits_d[:,1,:,:] # N(H+2)(W+2)C
        logits_d[:,-1,:,:] = logits_d[:,-2,:,:] # N(H+2)(W+2)C
        logits_d[:,:,0,:] = logits_d[:,:,1,:] # N(H+2)(W+2)C
        logits_d[:,:,-1,:] = logits_d[:,:,-2,:] # N(H+2)(W+2)C
        
        """
        | 4| 0| 5|
        | 2| 8| 3|
        | 6| 1| 7|
        """
        x_range = [1, -1,  0, 0, -1,  1, -1,  1, 0]
        y_range = [0,  0, -1, 1,  1,  1, -1, -1, 0]
        dist_maps = torch.zeros((0,len(x))).cuda() # 8k
        kl_maps = torch.zeros((0,len(x))).cuda() # 8k

        kl_center = logits[(n,x,y)] # KC
        for dx, dy in zip(x_range, y_range):
            dist_now = pred_dist_map_d[(n,x+dx+1,y+dy+1)]
            dist_maps = torch.cat((dist_maps,dist_now.unsqueeze(0)),0)

            if dx != 0 or dy != 0:
                logits_now = logits_d[(n,x+dx+1,y+dy+1)]
                # kl_map_now = torch.kl_div((kl_center+eps).log(), logits_now+eps).sum(2)  # 8KC->8K
                if self.isdetach:
                    logits_now = logits_now.detach()
                kl_map_now = kl_div(kl_center, logits_now)
                
                kl_map_now = kl_map_now.sum(1)  # KC->K
                kl_maps = torch.cat((kl_maps,kl_map_now.unsqueeze(0)),0)
                torch.clamp(kl_maps, min=0.0, max=20.0)

        # direction_gt shound be Nk  (8k->K)
        direction_gt = torch.argmin(dist_maps, dim=0)
        # weight_ce = pred_dist_map[bound]
        weight_ce = pred_dist_map[(n,x,y)]
        # print(weight_ce)

        # delete if min is 8 (local position)
        direction_gt_idx = [direction_gt!=8]
        direction_gt = direction_gt[direction_gt_idx]


        kl_maps = torch.transpose(kl_maps,0,1)
        direction_pred = kl_maps[direction_gt_idx]
        weight_ce = weight_ce[direction_gt_idx]

        return direction_gt, direction_pred, weight_ce

    def get_dist_maps(self, target):
        target_detach = target.clone().detach()
        dist_maps = torch.cat([self.dist_map_transform(target_detach[i]) for i in range(target_detach.shape[0])])
        out = -dist_maps
        out = torch.where(out>0, out, torch.zeros_like(out))
        
        return out

    def forward(self, logits, target, dist_maps, save=False):
        
        #ph, pw = logits.size(2), logits.size(3)
        #h, w = target.size(1), target.size(2)
        
        #if ph != h or pw != w:
        #    print("interpolated")
        #    logits = F.interpolate(input=logits, size=(
        #        h, w), mode='bilinear', align_corners=True)

        #gt_boundary = self.gt2boundary(target, ignore_label=self.ignore_label)

        #dist_maps = self.get_dist_maps(gt_boundary).cuda() # <-- it will slow down the training, you can put it to dataloader.

        pred_boundary = self.logits2boundary(logits, dist_maps)
        if save:
            plt.imshow(pred_boundary[0].cpu().numpy())
            plt.savefig(f"pred_boundary_{0}.png", dpi=150)
            plt.imshow(dist_maps[0].cpu().numpy())
            plt.savefig(f"dist_maps_{0}.png", dpi=150)
        
        save = False
        
        if pred_boundary.sum() < 1: # avoid nan
            return None # you should check in the outside. if None, skip this loss.
        
        direction_gt, direction_pred, weight_ce = self.get_direction_gt_predkl(dist_maps, pred_boundary, logits) # NHW,NHW,NCHW
        
        # direction_pred [K,8], direction_gt [K]
        loss = self.criterion(direction_pred, direction_gt) # careful
        
        weight_ce = self.weight_func(weight_ce)
        loss = (loss * weight_ce).mean()  # add distance weight

        return loss


dist_map_transform = transforms.Compose([
        lambda img: img.unsqueeze(0),
        lambda nd: nd.type(torch.int64),
        partial(class2one_hot, C=1),
        itemgetter(0),
        lambda t: t.cpu().numpy(),
        one_hot2dist,
        lambda nd: torch.tensor(nd, dtype=torch.float32)
    ])
    
def gt2boundary(gt, ignore_label=-1):  # gt NHW
    gt_lr = gt[:,1:,:]-gt[:,:-1,:]  # NHW
    gt_ud = gt[:,:,1:]-gt[:,:,:-1]
    gt_lr = torch.nn.functional.pad(gt_lr, [0,0,0,1,0,0], mode='constant', value=0) != 0 
    gt_ud = torch.nn.functional.pad(gt_ud, [0,1,0,0,0,0], mode='constant', value=0) != 0
    gt_combine = gt_lr+gt_ud
    del gt_lr
    del gt_ud

    # set 'ignore area' to all boundary
    gt_combine += (gt==ignore_label)

    return gt_combine > 0    
    
def get_dist_maps(target):
        target_detach = target.clone().detach()
        dist_maps = torch.cat([dist_map_transform(target_detach[i]) for i in range(target_detach.shape[0])])
        out = -dist_maps
        out = torch.where(out>0, out, torch.zeros_like(out))
        
        return out

def create_distmaps(target):
    h, w = target.shape[1], target.shape[2]
    
    gt_boundary = gt2boundary(target, ignore_label=255)
    
    dist_map = get_dist_maps(gt_boundary)
    
    return dist_map
    

if __name__ == '__main__':
    import os
    import random

    n,c,h,w = 1,2,100,100
    gt = torch.zeros((n,h,w)).cuda()
    gt[0,5] = 1
    gt[0,50] = 1
    logits = torch.randn((n,h,w,c)).cuda()

    abl = ABL(label_smoothing=0.5)
    print(abl(logits, gt))