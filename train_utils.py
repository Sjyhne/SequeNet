import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.gridspec import GridSpec

from models.metrics import boundary_iou

from models.label_smooth import LabelSmoothSoftmaxCEV1
from models.abl import ABL
from models.models.loss.rmi import RMILoss

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
        if key != "logits":
            if value != np.nan:
                metrics[key].append(value)
            else:
                print("Error:", key, "-", value)
    
    return metrics

def display_and_store_metrics(train, eval, args):

    print("Train loss: %.4f | Val loss: %.4f" % (float(train["loss"]), float(eval["loss"])))
    print("Train acc: %.4f | Val acc: %.4f" % (float(train["acc"]), float(eval["acc"])))
    print("Train miou: %.4f | Val miou: %.4f" % (float(train["miou"]), float(eval["miou"])))
    print("Train biou: %.4f | Val biou: %.4f" % (float(train["biou"]), float(eval["biou"])))
    print()
    
    store_path = args.output_path
    
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

class ABLLoss(torch.nn.Module):
    def __init__(self, abl_weight = 1.0):
        super(ABLLoss, self).__init__()
        self.cc = torch.nn.CrossEntropyLoss()
        self.lovasz_softmax = LabelSmoothSoftmaxCEV1()
        self.abl = ABL()
        self.abl_weight = abl_weight
    
    def forward(self, logits, targets, dist_maps, save=False):
        cc_loss = self.cc(logits, targets)
        lovasz_loss = self.lovasz_softmax(logits, targets)
        abl_loss = self.abl(logits, targets, dist_maps, save)
        
        
        if abl_loss == None:
            return cc_loss + lovasz_loss
        else:
            return cc_loss + (abl_loss * self.abl_weight) + lovasz_loss

def get_loss(args):
    if args.loss == "cce":
        return torch.nn.CrossEntropyLoss()
    if args.loss == "abl":
        return ABLLoss(abl_weight = args.abl_weight)
    if args.loss == "rmi":
        return RMILoss(num_classes=args.num_classes)
        
def get_optim(optim):
    if optim == "adam":
        return torch.optim.Adam
    elif optim == "sgd":
        return torch.optim.SGD

def calc_biou(pred_imgs, anns):
    biou = []
    for i, pi in enumerate(pred_imgs):
        ann = anns[i].cpu().numpy().astype(np.uint8)
        pred_img = pi.cpu().numpy().astype(np.uint8)
        biou.append(boundary_iou(ann, pred_img))
    
    return np.mean(biou)

def save_best_model(model, optim, loss_value, best_loss_value, epoch, args):
    if best_loss_value == None or loss_value < best_loss_value:
        path = args.output_path
        save_path = os.path.join(path, f"{args.model}_best_model_{epoch}.tar")
        torch.save({
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "epoch": epoch,
            "loss": loss_value,
        }, save_path)
        return loss_value
    else:
        return best_loss_value

def store_images(path, batch, pred_images):
    anns = batch["lab"]
    imgs = batch["orig_img"]
    if batch["orig_img"].shape[3] > 3:
        imgs = np.uint8(batch["orig_img"] * 255)
    names = batch["name"]
    os.makedirs(path)
    anns = anns.unsqueeze(-1)
    pred_images = torch.permute(pred_images, (0, 2, 3, 1))
    pred_images = torch.nn.functional.softmax(pred_images, dim=-1).cpu().detach().numpy()
    highlighted_images = np.uint8((np.float32(imgs[:, :, :, :3]) * np.expand_dims(np.clip(pred_images[:, :, :, 1], 0.2, 1.0), axis=-1)))
    main_build_gradients = np.expand_dims(pred_images[:, :, :, 1], axis=-1)
    pred_images = np.expand_dims(np.argmax(pred_images, axis=-1), axis=-1)
    for i, pi in enumerate(pred_images[:6]):
        fig = plt.figure(constrained_layout=True)
        gs = GridSpec(3, 2, figure=fig)
        fig.add_subplot(gs[0, 0])
        fig.add_subplot(gs[0, 1])
        fig.add_subplot(gs[1, 0])
        fig.add_subplot(gs[1, 1])
        fig.add_subplot(gs[2, 0])
        fig.axes[0].imshow(np.uint8(imgs[i][:, :, :3]))
        fig.axes[0].set_title("RGB")
        fig.axes[1].imshow(highlighted_images[i])
        fig.axes[1].set_title("Mask * RGB")
        fig.axes[2].imshow(anns[i])
        fig.axes[2].set_title("Ground Truth")
        fig.axes[3].imshow(pi)
        fig.axes[3].set_title("Predicted Mask")
        fig.axes[4].imshow(main_build_gradients[i])
        fig.axes[4].set_title("Build Gradients")
        
        plt.savefig(os.path.join(path, f"{names[i]}.png"), dpi=250)
        plt.close()

        
SMOOTH = 1e-6

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    #outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    #thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return iou.mean()  # Or thresholded.mean() if you are interested in average across the batch