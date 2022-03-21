#!/user/bin/python# -*- encoding: utf-8 -*-
import torch
import time 
from loss.tracingloss import tracingloss 
from utils import Averagvalue, save_checkpoint
import os 
from os.path import join, isdir
import torchvision

def train(cfg, train_loader, model, optimizer, scheduler, epoch, save_dir):    
    batch_time = Averagvalue()    
    data_time = Averagvalue()    
    losses = Averagvalue()    # switch to train mode    
    model.train()    
    end = time.time()    
    epoch_loss = []    
    counter = 0    
    for i, (image, label, pth) in enumerate(train_loader):        
        # measure data loading time        
        data_time.update(time.time() - end)        
        image, label = image.cuda(), label.cuda()        
        outputs = model(image)        
        loss = torch.zeros(1).cuda()        
        for o in outputs[:3]:            
            loss = loss + tracingloss(o, label, tex_factor=0.1, bdr_factor=2., balanced_w=1.1)        
            loss = loss + tracingloss(outputs[3], label, tex_factor=0.05, bdr_factor=1, balanced_w = 1.1)        
            loss = loss + tracingloss(outputs[4], label, tex_factor=0.05, bdr_factor=1, balanced_w = 1.1)        
            loss = loss + tracingloss(outputs[5], label, tex_factor=0.02, bdr_factor=4, balanced_w=1.1)        
            counter += 1        
            loss = loss / cfg.itersize        
            loss.backward()        
            if counter == cfg.itersize:            
                optimizer.step()            
                optimizer.zero_grad()            
                counter = 0        # measure accuracy and record loss        
                losses.update(loss.item(), image.size(0))        
                epoch_loss.append(loss.item())        
                batch_time.update(time.time() - end)        
                end = time.time()        # display and logging        
                if not isdir(save_dir):            
                    os.makedirs(save_dir)        
                    if i % cfg.msg_iter == 0:            
                        info = 'Epoch: [{0}/{1}][{2}/{3}] '.format(epoch, cfg.max_epoch, i, len(train_loader)) + \                   
                        'Time {batch_time.val:.3f} (avg:{batch_time.avg:.3f}) '.format(batch_time=batch_time) + \                   
                        'Loss {loss.val:f} (avg:{loss.avg:f}) '.format(                       
                            loss=losses)            
                        print(info)            
                        label_out = torch.eq(label, 1).float()            
                        outputs.append(label_out)            
                        _, _, H, W = outputs[0].shape            
                        all_results = torch.zeros((len(outputs), 1, H, W))            
                        for j in range(len(outputs)):                
                            all_results[j, 0, :, :] = outputs[j][0, 0, :, :]            
                            torchvision.utils.save_image(1-all_results, join(save_dir, "iter-%d.jpg" % i))    # adjust lr    
                            scheduler.step()    # save checkpoint    
                            save_checkpoint({       
                                'epoch': epoch,       
                                'state_dict': model.state_dict(),       
                                'optimizer': optimizer.state_dict()           
                            }, filename=join(save_dir, "epoch-%d-checkpoint.pth" % epoch))
                            
                            return losses.avg, epoch_loss
