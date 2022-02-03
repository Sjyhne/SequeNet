import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.gridspec import GridSpec

import os
import shutil
import csv

from models.metrics import boundary_iou

params = {'legend.fontsize': 'xx-small',
         'axes.labelsize': 'xx-small',
         'axes.titlesize':'xx-small',
         'xtick.labelsize':'xx-small',
         'ytick.labelsize':'xx-small'}
pylab.rcParams.update(params)


def remove_all_folders_in_path(path):
    for folder in os.listdir(path):
        if os.path.isdir(os.path.join(path, folder)):
            shutil.rmtree(os.path.join(path, folder))

def store_images(path, anns, imgs, pred_images, extra_pred_images=None):
    os.makedirs(path)
    if extra_pred_images == None:
        highlighted_images = (tf.cast(imgs, dtype=tf.float32) * tf.expand_dims(tf.clip_by_value(pred_images[:, :, :, 1], 0.2, 0.9), axis=-1))/255
        pred_images = tf.expand_dims(tf.math.argmax(pred_images, axis=-1), axis=-1)
        for i, pi in enumerate(pred_images):
            fig = plt.figure(constrained_layout=True)
            gs = GridSpec(2, 2, figure=fig)
            fig.add_subplot(gs[0, 0])
            fig.add_subplot(gs[0, 1])
            fig.add_subplot(gs[1, 0])
            fig.add_subplot(gs[1, 1])
        
            fig.axes[0].imshow(imgs[i])
            fig.axes[1].imshow(anns[i])
            fig.axes[2].imshow(highlighted_images[i])
            fig.axes[3].imshow(pi)
            
            plt.savefig(os.path.join(path, f"{i}_.png"), dpi=200)
    else:    
        highlighted_images = (tf.cast(imgs, dtype=tf.float32) * tf.expand_dims(tf.clip_by_value(pred_images[:, :, :, 1], 0.2, 0.9), axis=-1))/255
        extra_highlighted_images = (tf.cast(imgs, dtype=tf.float32) * tf.expand_dims(tf.clip_by_value(extra_pred_images[:, :, :, 1], 0.2, 0.9), axis=-1))/255
        pred_images = tf.expand_dims(tf.math.argmax(pred_images, axis=-1), axis=-1)
        extra_pred_images = tf.expand_dims(tf.math.argmax(extra_pred_images, axis=-1), axis=-1)
        
        for i, pi in enumerate(pred_images):
            fig = plt.figure(constrained_layout=True)
            gs = GridSpec(2, 3, figure=fig)
            fig.add_subplot(gs[0, 0])
            fig.add_subplot(gs[0, 1])
            fig.add_subplot(gs[0, 2])
            fig.add_subplot(gs[1, 0])
            fig.add_subplot(gs[1, 1])
            fig.add_subplot(gs[1, 2])
        
            fig.axes[0].imshow(imgs[i])
            fig.axes[0].set_title("RGB")
            fig.axes[1].imshow(highlighted_images[i])
            fig.axes[1].set_title("Mask * RGB")
            fig.axes[2].imshow(extra_highlighted_images[i])
            fig.axes[2].set_title("XMask * RGB")
            fig.axes[3].imshow(anns[i])
            fig.axes[3].set_title("Ground Truth")
            fig.axes[4].imshow(pi)
            fig.axes[4].set_title("Mask")
            fig.axes[5].imshow(extra_pred_images[i])
            fig.axes[5].set_title("XMask")
            
            plt.savefig(os.path.join(path, f"{i}_.png"), dpi=200)
            plt.close()
        

def calc_biou(pred_imgs, anns):
    biou = []
    for i, pi in enumerate(pred_imgs):
        ann = anns[i].numpy().astype("uint8")
        pred_img = pi.numpy().astype("uint8")
        biou.append(boundary_iou(ann, pred_img))
    
    return np.mean(biou)

def save_best_model(model, loss_value, best_loss_value, epoch):
    if loss_value < best_loss_value:
        save_path = os.path.join("model_output", f"{epoch}_.h5")
        model.save(save_path)
        return loss_value
    else:
        return best_loss_value
    
def calculate_sample_weight(labels, num_classes):
    unique_values = range(num_classes)
    unique_value_count = []
    sample_weights = []
    for b in labels:
        b_count = []
        b_total = 0
        for val in unique_values:
            count = tf.math.count_nonzero(tf.math.equal(b, val)).numpy()
            b_total += count
            b_count.append(count)
        sample_weights.append(b_count[1] / b_total)
    return sample_weights
        

def display_and_store_metrics(tlm, vlm, tmm, vmm, tbm, vbm, name):

    train_loss = np.mean(tlm)
    val_loss = np.mean(vlm)
    train_miou = np.mean(tmm)
    val_miou = np.mean(vmm)
    train_biou = np.mean(tbm)
    val_biou = np.mean(vbm)

    print("Train loss: %.4f | Val loss: %.4f" % (float(train_loss), float(val_loss)))
    print("Train miou: %.4f | Val miou: %.4f" % (float(train_miou), float(val_miou)))
    print("Train biou: %.4f | Val biou: %.4f" % (float(train_biou), float(val_biou)))
    print()

    fpath = f"logs/{name}_metrics.csv"

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
