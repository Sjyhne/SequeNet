import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.gridspec import GridSpec

from models.metrics import boundary_iou

import os
import shutil
import csv

params = {'legend.fontsize': 'xx-small',
         'axes.labelsize': 'xx-small',
         'axes.titlesize':'xx-small',
         'xtick.labelsize':'xx-small',
         'ytick.labelsize':'xx-small'}
pylab.rcParams.update(params)

def aggregate_metrics(metrics, result):
    for key, value in result.items():
        metrics[key].append(value)
    
    return metrics

def display_and_store_metrics(train, eval, args):

    print("Train loss: %.4f | Val loss: %.4f" % (float(train["loss"]), float(eval["loss"])))
    print("Train acc: %.4f | Val acc: %.4f" % (float(train["acc"]), float(eval["acc"])))
    print("Train miou: %.4f | Val miou: %.4f" % (float(train["miou"]), float(eval["miou"])))
    print("Train biou: %.4f | Val biou: %.4f" % (float(train["biou"]), float(eval["biou"])))
    print()
    
    store_path = os.path.join("model_output", f"{args.training_mode}_{args.model}")
    
    if not os.path.exists(store_path):
        os.makedirs(store_path)

    fpath = os.path.join(store_path, "metrics.csv")

    losses = [
        train["loss"], eval["loss"],
        train["acc"], eval["acc"],
        train["miou"], eval["miou"],
        train["biou"], eval["biou"]
    ]

    if os.path.exists(fpath):
        with open(fpath, "a") as f:
            writer = csv.writer(f)
            writer.writerow(losses)
    else:
        with open(fpath, "a") as f:
            writer = csv.writer(f)
            headers = [
                "train_loss", "val_loss",
                "train_acc", "val_acc",
                "train_miou", "val_miou",
                "train_biou", "val_biou"
            ]
            writer.writerow(headers)
            writer.writerow(losses)

def get_loss(loss):
    if loss == "cce":
        return torch.nn.CrossEntropyLoss

def get_optim(optim):
    if optim == "adam":
        return torch.optim.Adam

def calc_biou(pred_imgs, anns):
    biou = []
    for i, pi in enumerate(pred_imgs):
        ann = anns[i].numpy().astype("uint8")
        pred_img = pi.numpy().astype("uint8")
        biou.append(boundary_iou(ann, pred_img))
    
    return np.mean(biou)

def save_best_model(model, loss_value, best_loss_value, epoch, args):
    if best_loss_value == None or loss_value < best_loss_value:
        path = os.path.join("model_output", f"{args.training_mode}_{args.model}")
        for file in os.listdir(path):
            if "best_model" in file:
                shutil.rmtree(os.path.join(path, file))
        save_path = os.path.join(path, f"{args.model}_best_model_{epoch}.pt")
        torch.save(model.state_dict(), save_path)
        return loss_value
    else:
        return best_loss_value

def store_images(path, anns, imgs, pred_images, names):
    os.makedirs(path)
    print(anns.shape)
    print(imgs.shape)
    print(pred_images.shape)
    anns = torch.permute(anns, (0, 2, 3, 1))
    imgs = torch.permute(imgs, (0, 2, 3, 1))
    pred_images = torch.permute(pred_images, (0, 2, 3, 1))
    print(anns.shape)
    print(imgs.shape)
    print(pred_images.shape)
    highlighted_images = (imgs.float() * torch.unsqueeze(torch.clip(pred_images[:, :, :, 1], 0.2, 1.0), dim=1))/255
    main_build_gradients = torch.unsqueeze(pred_images[:, :, :, 1], dim=-1)
    pred_images = torch.unsqueeze(torch.argmax(pred_images, dim=-1), dim=-1)
    for i, pi in enumerate(pred_images):
        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(3, 2, figure=fig)
        fig.add_subplot(gs[0, 0])
        fig.add_subplot(gs[0, 1])
        fig.add_subplot(gs[1, 0])
        fig.add_subplot(gs[1, 1])
        fig.add_subplot(gs[2, 0])
    
        fig.axes[0].imshow(imgs[i])
        fig.axes[0].set_title("RGB")
        fig.axes[1].imshow(highlighted_images[i])
        fig.axes[1].set_title("Mask * RGB")
        fig.axes[2].imshow(anns[i])
        fig.axes[2].set_title("Ground Truth")
        fig.axes[3].imshow(pi)
        fig.axes[3].set_title("Mask")
        fig.axes[4].imshow(main_build_gradients[i])
        fig.axes[4].set_title("Build Gradients")
        
        plt.savefig(os.path.join(path, f"{names[i]}.png"), dpi=200)
        plt.close()