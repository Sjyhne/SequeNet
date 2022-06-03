import numpy as np
import cv2 as cv
from tqdm import tqdm
import matplotlib.pyplot as plt

from generator import create_dataset_generator
from models.models.cats import Network
from cats_main import Config

from models.abl import gt2boundary

import torch

import argparse
import os
import shutil

def dilate_masks(masks, dilation=(10, 10)):
    
    dilation = np.ones((dilation))
    
    dilated_masks = np.ndarray(masks.shape)
    
    masks = np.uint8(masks)
    
    for i, mask in enumerate(masks):
        mask = cv.dilate(mask, dilation)
        dilated_masks[i] = mask
    
    return dilated_masks

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Add custom arguments for the training of the model(s)")
    # TODO: This should be revised -- maybe added in a config file or something
    parser.add_argument("--dn", type=str, help="Path of the dataset up for generation")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--num_channels", type=int, default=3, help="Number of channels in the input image")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes in the input image")
    parser.add_argument("--image_dim", type=int, default=512, help="Input image dimension")
    parser.add_argument("--four_channels", type=bool, default=True, help="Whether the gradients should be put as a 4th channel")
    
    args = parser.parse_args()
    
    print("args:", args)
    
    original_datapath = os.path.join("data", args.dn)
    
    label_path = "data/large_building_area"
    
    train_ds = torch.utils.data.DataLoader(create_dataset_generator(original_datapath, "train", data_percentage=1.0), shuffle=False, batch_size=args.bs)
    val_ds = torch.utils.data.DataLoader(create_dataset_generator(original_datapath, "val", data_percentage=1.0), shuffle=False, batch_size=args.bs)
    test_ds = torch.utils.data.DataLoader(create_dataset_generator(original_datapath, "test", data_percentage=1.0), shuffle=False, batch_size=args.bs)
    
    
    if args.four_channels:
        new_datapath = os.path.join("data", "perfect_edge_dataset" + "_" + args.dn)
        if os.path.exists(new_datapath):
            shutil.rmtree(new_datapath)
    else:
        new_datapath = os.path.join("data", "perfect_edge_dataset" + "_" + args.dn)
        if os.path.exists(new_datapath):
            shutil.rmtree(new_datapath)
    
    os.makedirs(new_datapath)
    
    print("Original datapath:", original_datapath)
    print("Destinaion datapath:", new_datapath)
    
    types = ["test", "val", "train"]
    imagetypes = ["ann_dir", "img_dir", "mask_dir", "grad_dir"]
    
    for t in types:
        for it in imagetypes:
            os.makedirs(os.path.join(new_datapath, t, it))
    
    data = [test_ds, val_ds, train_ds]
    
    if not args.four_channels:
        for i, d in enumerate(data):
            for step, batch in tqdm(enumerate(d), total=len(d)):
                img_dir = os.path.join(new_datapath, types[i], "img_dir")
                ann_dir = os.path.join(new_datapath, types[i], "ann_dir")
                mask_dir = os.path.join(new_datapath, types[i], "mask_dir")
                grad_dir = os.path.join(new_datapath, types[i], "grad_dir")

                names = batch["name"]
                dist_map = batch["dist_map"].cpu().numpy()
                orig_imgs = batch["orig_img"].cpu().numpy()
                imgs = batch["img"].to(args.device).permute(0, 3, 1, 2)
                labels = batch["lab"].to(args.device).cpu().numpy()

                lpaths = [os.path.join(label_path, types[i], "ann_dir", n + ".tiff") for n in names]
                
                predictions = np.asarray([cv.imread(lp, cv.IMREAD_GRAYSCALE) for lp in lpaths])
                
                predicted_masks = gt2boundary(torch.tensor(predictions))
                
                for it, pm in enumerate(predicted_masks):
                    plt.imsave(mask_dir + "/" + names[it] + ".png", pm)

                gradients = predicted_masks.numpy()
                for it, grad in enumerate(gradients):
                    plt.imsave(grad_dir + "/" + names[it] + ".png", grad)

                for it, ann in enumerate(labels):
                    plt.imsave(ann_dir + "/" + names[it] + ".png", ann)
                    np.save(ann_dir + "/" + names[it] + ".npy", dist_map[i])

                dilated_masks = dilate_masks(predicted_masks, (8, 8))
                dilation_grads = np.clip(gradients + 0.7, 0.0, 1.0)
                gradient_masks = np.clip(np.expand_dims(dilated_masks * dilation_grads, axis=-1), 0.2, 1.0)
                gradient_imgs = np.uint8(orig_imgs * gradient_masks)

                for it, im in enumerate(gradient_imgs):
                    plt.imsave(img_dir + "/" + names[it] + ".png", im)
                    
                exit()
                """    
                plt.imsave(f"{names[0]}_distmap.png", dist_map[0])
                plt.imsave(f"{names[0]}_label.png", labels[0])
                plt.imsave(f"{names[0]}_predmask.png", predicted_masks[0])
                plt.imsave(f"{names[0]}_predgrad.png", gradients[0])
                plt.imsave(f"{names[0]}_dilated_predmask.png", dilated_masks[0])
                plt.imsave(f"{names[0]}_increased_grads.png", dilation_grads[0])
                plt.imsave(f"{names[0]}_gradient_mask.png", np.squeeze(gradient_masks, axis=-1)[0])
                plt.imsave(f"{names[0]}_final_image.png", gradient_imgs[0])
                plt.imsave(f"{names[0]}_original_image.png", orig_imgs[0])
                """
    else:
        for i, d in enumerate(data):
            for step, batch in tqdm(enumerate(d), total=len(d)):
                img_dir = os.path.join(new_datapath, types[i], "img_dir")
                ann_dir = os.path.join(new_datapath, types[i], "ann_dir")
                mask_dir = os.path.join(new_datapath, types[i], "mask_dir")
                grad_dir = os.path.join(new_datapath, types[i], "grad_dir")

                names = batch["name"]
                dist_map = batch["dist_map"].cpu().numpy()
                orig_imgs = batch["orig_img"].cpu().numpy()
                imgs = batch["img"].to(args.device).permute(0, 3, 1, 2)
                labels = batch["lab"].to(args.device).cpu().numpy()
                
                lpaths = [os.path.join(label_path, types[i], "edge_dir", n + ".png") for n in names]

                predictions = np.asarray([cv.imread(lp, cv.IMREAD_GRAYSCALE) for lp in lpaths])
                
                predicted_masks = gt2boundary(torch.tensor(predictions))
                
                dilated_masks = dilate_masks(predicted_masks, (8, 8))
                
                dilated_masks = np.expand_dims(dilated_masks, axis=-1)
                
                predictions[predictions == 30] = 0.0
                predictions[predictions == 215] = 1.0
                
                predictions = np.clip(predictions, 0.5, 1.0)
                
                predictions = np.expand_dims(predictions, axis=-1)
                
                new = (orig_imgs/255) * predictions
                
                #new = np.concatenate((orig_imgs/255, dilated_masks), axis=-1)
                for it, n in enumerate(new):
                    plt.imsave(img_dir + "/" + names[it] + ".png", np.uint8(n * 255))
                
