from models.abl import create_distmaps

import numpy as np
import cv2 as cv
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch

import argparse
import os

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Add custom arguments for the training of the model(s)")
    # TODO: This should be revised -- maybe added in a config file or something
    parser.add_argument("--dn", type=str, help="Name of dataset to create distance maps for")

    args = parser.parse_args()
    
    path = os.path.join("data", args.dn)
    
    folders = os.listdir(path)[1:]
    
    image_folders = [os.path.join(path, folder, "ann_dir") for folder in folders]
    
    for folder_path in image_folders:
        print("Folder path:", folder_path)
        for _, file in tqdm(enumerate(os.listdir(folder_path)), total=len(os.listdir(folder_path)), leave=False):
            if file.split(".")[-1] in ["tiff", "tif", "png"]:
                filepath = os.path.join(folder_path, file)
                target = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
                target = torch.tensor(np.expand_dims(target, axis=0))
                dist_map = create_distmaps(target).numpy()
                dist_path = filepath.split(".")[0] + ".npy"
                if dist_map.shape == 4:
                    dist_map = np.squeeze(dist_map)
                if np.amax(dist_map) > 256:
                    dist_map = np.uint16(dist_map)
                else:
                    dist_map = np.uint8(dist_map)
                
                np.save(dist_path, dist_map)
    