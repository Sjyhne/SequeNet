#!/user/bin/python
# -*- encoding: utf-8 -*-

import os
import torch
import torchvision
from PIL import Image
from os.path import join, isdir
import numpy as np
from tqdm import tqdm
import cv2

from models.models.cats_loss import tracingloss

def test(cfg, model, test_loader, save_dir):
    model.eval()
    dl = tqdm(test_loader, leave=False)
    if not isdir(save_dir):
        os.makedirs(save_dir)
        
    loss = 0
    
    iou = 0
    
    for batch in dl:
        dl.set_description("Single-scale test")
        image = torch.permute(batch["img"][0].cuda().unsqueeze(0), (0, 3, 1, 2))
        label = batch["lab"].cuda()
        dist_map = batch["dist_map"].cuda()
        
        _, _, H, W = image.shape
        filename = batch["name"][0]
        results = model(image)
        for o in results[:3]:
            loss = loss + tracingloss(o, label, dist_map, tex_factor=0.1, bdr_factor=2., balanced_w=1.1).item()
        loss = loss + tracingloss(results[3], label, dist_map, tex_factor=0.05, bdr_factor=1, balanced_w = 1.1).item()
        loss = loss + tracingloss(results[4], label, dist_map, tex_factor=0.05, bdr_factor=1, balanced_w = 1.1).item()
        loss = loss + tracingloss(results[5], label, dist_map, tex_factor=0.02, bdr_factor=4, balanced_w=1.1).item()
        results = [torch.sigmoid(r) for r in results]
        if cfg.side_edge:
            results_all = torch.zeros((len(results), 1, H, W))
            for i in range(len(results)):
                results_all[i, 0, :, :] = results[i]
            torchvision.utils.save_image((1-results_all), join(save_dir, "%s.jpg" % filename))

        result = torch.squeeze(results[-1].detach()).cpu().numpy()
        result[result < 0.5] = 0
        result[result >= 0.5] = 1
        label = np.squeeze(label.detach().cpu().numpy())
        overlap = result * label
        union = result + label
        
        tempiou = overlap.sum()/float(union.sum())
        
        if np.isnan(tempiou):
            print("Fixed")
            tempiou = 1e-6
        
        iou += tempiou
        
        result = Image.fromarray((result * 255).astype(np.uint8))
        result.save(join(save_dir, "%s.png" % filename))
    
    print("Total eval loss:", loss / len(dl))
    print("Total IoU:", iou / len(dl))



def multiscale_test(model, test_loader, save_dir):
    model.eval()
    dl = tqdm(test_loader)
    if not isdir(save_dir):
        os.makedirs(save_dir)
    scale = [0.5, 1, 1.5]
    for batch in dl:
        
        image = torch.permute(batch["img"][0].cuda().unsqueeze(0), (0, 3, 1, 2))
        pth = batch["name"]
        
        dl.set_description("Single-scale test")
        image_in = image[0].cpu().numpy().transpose((1,2,0))
        _, H, W = image[0].shape
        multi_fuse = np.zeros((H, W), np.float32)
        for k in range(0, len(scale)):
            im_ = cv2.resize(image_in, None, fx=scale[k], fy=scale[k], interpolation=cv2.INTER_LINEAR)
            im_ = im_.transpose((2,0,1))
            results = model(torch.unsqueeze(torch.from_numpy(im_).cuda(), 0))
            result = torch.squeeze(results[-1].detach()).cpu().numpy()
            fuse = cv2.resize(result, (W, H), interpolation=cv2.INTER_LINEAR)
            multi_fuse += fuse
        multi_fuse = multi_fuse / len(scale)
        ### rescale trick suggested by jiangjiang
        multi_fuse = (multi_fuse - multi_fuse.min()) / (multi_fuse.max() - multi_fuse.min())
        filename = pth[0]
        result_out = Image.fromarray(((1-multi_fuse) * 255).astype(np.uint8))
        result_out.save(join(save_dir, "%s.jpg" % filename))
        result_out_test = Image.fromarray((multi_fuse * 255).astype(np.uint8))
        result_out_test.save(join(save_dir, "%s.png" % filename))