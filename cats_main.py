#!/user/bin/python
# -*- encoding: utf-8 -*-

import os, sys
import numpy as np
from PIL import Image
import cv2
import shutil
import argparse
import time
import datetime
import torch
from generator import create_dataset_generator
from models.models.cats import Network
from models.models.cats_optimizer import Optimizer
from torch.utils.data import DataLoader, sampler
from cats_utils import Logger, Averagvalue, save_checkpoint
from os.path import join, split, isdir, isfile, splitext, split, abspath, dirname
from cats_train import train, DEVICE
from cats_test import test, multiscale_test
from models.models.cats_loss import DEVICE

class Config(object):
    def __init__(self):
        self.data = "inria_not_pretrained"
        # ============== training
        self.resume = "output/inria_not_pretrained/train/epoch-38-training-record/epoch-38-checkpoint.pth"
        self.msg_iter = 20
        self.gpu = '0'
        self.save_pth = join("./output", self.data)
        self.pretrained = "cats_pretrained/not_found.pth"
        self.aug = True

        # ============== testing
        self.multi_aug = False # Produce the multi-scale results
        self.side_edge = False # Output the side edges

        # ================ dataset
        self.dataset = "./data/{}".format(self.data)

        # =============== optimizer
        self.batch_size = 1
        self.lr = 1e-3
        self.momentum = 0.9
        self.wd = 2e-4
        self.stepsize = 5
        self.gamma = 0.1
        self.max_epoch = 40
        self.itersize = 16

def main():
    
    parser = argparse.ArgumentParser(description='Mode Selection')
    parser.add_argument('--mode', default = 'test', type = str,required=True, choices={"train", "test"}, help = "Setting models for training or testing")
    parser.add_argument('--data_path', default = 'data/inria_aerial_image_dataset', type = str)
    parser.add_argument('--batch_size', default = 8, type = int)
    parser.add_argument('--data_percentage', default = 1.0, type = float)
    parser.add_argument('--four_channels', default = False, type = bool)
    parser.add_argument('--transform', default = False, type = bool)

    args = parser.parse_args()

    cfg = Config()

    cfg.batch_size = args.batch_size

    if cfg.gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu

    THIS_DIR = abspath(dirname(__file__))
    TMP_DIR = join(THIS_DIR, cfg.save_pth)

    if not isdir(TMP_DIR):
        os.makedirs(TMP_DIR)
    
    # model
    model = Network(cfg)
    print('=> Load model')

    model.to(DEVICE)
    print('=> Cuda used')
    
    train_ds = torch.utils.data.DataLoader(create_dataset_generator(args.data_path, "train", batch_size=args.batch_size, data_percentage=args.data_percentage, four_channels=args.four_channels, transform=args.transform), shuffle=True, batch_size=args.batch_size)
    val_ds = torch.utils.data.DataLoader(create_dataset_generator(args.data_path, "val", batch_size=1, data_percentage=args.data_percentage, four_channels=args.four_channels), shuffle=False, batch_size=1)
    test_ds = torch.utils.data.DataLoader(create_dataset_generator(args.data_path, "test", batch_size=1, data_percentage=args.data_percentage, four_channels=args.four_channels), shuffle=False, batch_size=1)
    
    cfg.msg_iter = len(train_ds) - 1
    
    if args.mode == "test":
        assert isfile(cfg.resume), "No checkpoint is found at '{}'".format(cfg.resume)
        
        model.load_checkpoint()

        test(cfg, model, val_ds, save_dir = join(TMP_DIR, "test", "sing_scale_test"))

        if cfg.multi_aug:
            multiscale_test(model, test_ds, save_dir = join(TMP_DIR, "test", "multi_scale_test"))
    
    elif args.mode == "produce":
        assert isfile(cfg.resume), "No checkpoint is found at '{}'".format(cfg.resume)
        
        model.load_checkpoint()
        
        ds = [train_ds, val_ds, test_ds]
        dtype = ["train", "val", "test"]
        
    
    else:
        model.init_weight()

        if cfg.resume:
            model.load_checkpoint()

        model.train()

        # optimizer
        optim, scheduler = Optimizer(cfg)(model)

        # log
        log = Logger(join(TMP_DIR, "%s-%d-log.txt" %("sgd",cfg.lr)))
        sys.stdout = log

        train_loss = []
        train_loss_detail = []

        for epoch in range(0, cfg.max_epoch):
            
            print("Epoch", epoch, "/", cfg.max_epoch)

            tr_avg_loss, tr_detail_loss = train(cfg,
                train_ds, model, optim, scheduler, epoch,
                save_dir = join(TMP_DIR, "train", "epoch-%d-training-record" % epoch))

            test(cfg, model, val_ds, save_dir = join(TMP_DIR, "train", "epoch-%d-testing-record-view" % epoch))

            log.flush()

            train_loss.append(tr_avg_loss)
            train_loss_detail += tr_detail_loss

if __name__ == '__main__':
    main()