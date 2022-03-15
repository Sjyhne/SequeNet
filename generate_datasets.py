import numpy as np
import cv2 as cv
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.model_hub import get_model
from generator import create_dataset_generator

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
    parser.add_argument("--model", type=str, help="Name of model to datasets for")
    parser.add_argument("--model_type", type=str, help="Name of the model architecture used")
    parser.add_argument("--dn", type=str, help="Path of the dataset up for generation")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--bs", type=int, default=32)
    
    args = parser.parse_args()

    model = get_model(args.model_type)
    
    path = os.path.join("model_output", args.model)
    
    state_dict_name = None
    
    for file in os.listdir(path):
        if ".pt" in file:
            state_dict_name = file
    
    print(state_dict_name)
    
    model_state_path = os.path.join(path, state_dict_name)
    
    print(model_state_path)
    
    try:
        model.load_state_dict(torch.load(model_state_path))
        print("Successfully loaded model from state dict")
    except Exception as e:
        print(e)
    
    original_datapath = os.path.join("data", args.dn)
    
    train_ds = torch.utils.data.DataLoader(create_dataset_generator(original_datapath, "train", data_percentage=1.0), shuffle=False, batch_size=args.bs)
    val_ds = torch.utils.data.DataLoader(create_dataset_generator(original_datapath, "val", data_percentage=1.0), shuffle=False, batch_size=args.bs)
    test_ds = torch.utils.data.DataLoader(create_dataset_generator(original_datapath, "test", data_percentage=1.0), shuffle=False, batch_size=args.bs)
    
    new_datapath = os.path.join("data", args.model + "_" + args.dn)
    if os.path.exists(new_datapath):
        shutil.rmtree(new_datapath)
    
    os.makedirs(new_datapath)
    
    types = ["val", "test"]
    imagetypes = ["ann_dir", "img_dir", "mask_dir", "grad_dir"]
    
    for t in types:
        for it in imagetypes:
            os.makedirs(os.path.join(new_datapath, t, it))
    
    data = [val_ds, test_ds]
    
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
            
            predictions = torch.nn.functional.softmax(model(imgs), dim=1).cpu().detach()

            predicted_masks = torch.argmax(predictions, dim=1).cpu().detach().numpy()
            for it, pm in enumerate(predicted_masks):
                plt.imsave(mask_dir + "/" + names[it] + ".png", pm)
            
            gradients = predictions[:, 1, :, :].numpy()
            for it, grad in enumerate(gradients):
                plt.imsave(grad_dir + "/" + names[it] + ".png", grad)
            
            for it, ann in enumerate(labels):
                plt.imsave(ann_dir + "/" + names[it] + ".png", ann)
                np.save(ann_dir + "/" + names[it] + ".npy", dist_map[i])
            
            dilated_masks = dilate_masks(predicted_masks, (15, 15))
            dilation_grads = np.clip(gradients + 0.4, 0.0, 1.0)
            gradient_masks = np.clip(np.expand_dims(dilated_masks * dilation_grads, axis=-1), 0.15, 1.0)
            gradient_imgs = np.uint8(orig_imgs * gradient_masks)
            
            for it, im in enumerate(gradient_imgs):
                plt.imsave(img_dir + "/" + names[it] + ".png", im)
            
            
            