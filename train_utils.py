import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import os
import shutil
import csv

from models.metrics import boundary_iou

def remove_all_folders_in_path(path):
    for folder in os.listdir():
        if os.path.isdir(os.path.join(path, folder)):
            shutil.rmtree(os.path.join(path, folder))

def store_images(pred_images, anns, imgs, path):
    os.mkdir(path)
    for i, pred_img in enumerate(pred_images):
        f, x = plt.subplots(1, 3)
        x[0].imshow(pred_img.numpy())
        x[1].imshow(anns[i].numpy())
        x[2].imshow(imgs[i].numpy())
        plt.savefig(os.path.join(path, f"{i}_.png"), dpi=200)

def calc_biou(pred_imgs, anns):
    biou = []
    for i, pi in enumerate(pred_imgs):
        ann = anns[i].numpy().astype("uint8")
        pred_img = pi.numpy().astype("uint8")
        print(ann.shape, pred_img.shape)
        biou.append(boundary_iou(ann, pred_img))
    
    return np.mean(biou)

def display_and_store_metrics(tlm, vlm, tmm, vmm, tbm, vbm):

    mtrcs = {
        "train_loss": [], 
        "val_loss": [],
        "train_miou": [],
        "val_miou": [],
        "train_biou": [],
        "val_biou": []
        }

    train_loss = tlm.result()
    val_loss = vlm.result()
    train_miou = np.mean(tmm)
    val_miou = np.mean(vmm)
    train_biou = np.mean(tbm)
    val_biou = np.mean(vbm)

    mtrcs["train_loss"].append()

    print("Train loss: %.4f | Val loss: %.4f" % (float(train_loss), float(val_loss)))
    print("Train miou: %.4f | Val miou: %.4f" % (float(train_miou), float(val_miou)))
    print("Train biou: %.4f | Val biou: %.4f" % (float(train_biou), float(val_biou)))
    print()

    fpath = "logs/metrics.csv"

    losses = [
        train_loss, val_loss,
        train_miou, val_miou,
        train_biou, val_biou]

    if os.path.exists(fpath):
        with open(fpath, "a") as f:
            writer = csv.writer(f)
            writer.writerow(losses)
    else:
        with open(fpath, "a") as f:
            writer = csv.writer(f)
            headers = [
                "train_loss", "val_loss",
                "train_miou", "val_miou",
                "train_biou", "val_biou"
            ]
            writer.writerow(headers)
            writer.writerow(losses)
