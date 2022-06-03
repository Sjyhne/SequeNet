#!/user/bin/python
# -*- encoding: utf-8 -*-

import torch
import time
from models.models.cats_loss import tracingloss, DEVICE
from cats_utils import Averagvalue, save_checkpoint
import os
from os.path import join, isdir
import torchvision

from tqdm import tqdm

def train(cfg, train_loader, model, optimizer, scheduler, epoch, save_dir):
    batch_time = Averagvalue()
    data_time = Averagvalue()
    losses = Averagvalue()
    # switch to train mode
    model.train()
    end = time.time()
    epoch_loss = []
    counter = 0
    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):

        # measure data loading time
        data_time.update(time.time() - end)
        image, label = batch["img"].to(DEVICE), batch["lab"].to(DEVICE)
        dist_map = batch["dist_map"].to(DEVICE)
        
        image = torch.permute(image, (0, 3, 1, 2))
        #label = label.unsqueeze(1)
    
        outputs = model(image)
        loss = torch.zeros(1).to(DEVICE)
        
        for o in outputs[:3]:
            loss = loss + tracingloss(o, label, dist_map, tex_factor=0.1, bdr_factor=2., balanced_w=1.1)
        loss = loss + tracingloss(outputs[3], label, dist_map, tex_factor=0.05, bdr_factor=1, balanced_w = 1.1)
        loss = loss + tracingloss(outputs[4], label, dist_map, tex_factor=0.05, bdr_factor=1, balanced_w = 1.1)
        loss = loss + tracingloss(outputs[5], label, dist_map, tex_factor=0.02, bdr_factor=4, balanced_w=1.1)

        counter += 1
        loss = loss / cfg.batch_size
        loss.backward()

        if counter == cfg.itersize:
            optimizer.step()
            optimizer.zero_grad()
            counter = 0

        # measure accuracy and record loss
        losses.update(loss.item(), image.size(0))
        epoch_loss.append(loss.item())
        batch_time.update(time.time() - end)
        end = time.time()

        # display and logging
        if not isdir(save_dir):
            os.makedirs(save_dir)

        if i % cfg.msg_iter == 0 and i != 0:
            info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, cfg.max_epoch, i, len(train_loader)) + \
                   'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \
                   'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(
                       loss=losses)
            print(info)
            
            label_out = torch.eq(label, 1).float().unsqueeze(1)

            outputs.append(label_out)
            _, _, H, W = outputs[0].shape
            all_results = torch.zeros((len(outputs), 1, H, W))
            for j in range(len(outputs)):
                all_results[j, 0, :, :] = outputs[j][0, 0, :, :]
            torchvision.utils.save_image(1-all_results, join(save_dir, "iter-%d.jpg" % i))

    # adjust lr
    #scheduler.step()
    # save checkpoint
    save_checkpoint({
       'epoch': epoch,
       'state_dict': model.state_dict(),
       'optimizer': optimizer.state_dict()
           }, filename=join(save_dir, "epoch-%d-checkpoint.pth" % epoch))

    return losses.avg, epoch_loss