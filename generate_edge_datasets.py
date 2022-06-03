import numpy as np
import cv2 as cv
from tqdm import tqdm
import matplotlib.pyplot as plt

from generator import create_dataset_generator
from models.models.cats import Network
from cats_main import Config

import torch

import argparse
import os
import shutil

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Add custom arguments for the training of the model(s)")
    # TODO: This should be revised -- maybe added in a config file or something
    parser.add_argument("--cp", type=str, help="Path for checkpoint to be used")
    parser.add_argument("--epoch", type=int, help="Which epoch of the trained models should be used?")
    parser.add_argument("--dn", type=str, help="Path of the dataset up for generation")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--bs", type=int, default=8)
    parser.add_argument("--num_channels", type=int, default=3, help="Number of channels in the input image")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of classes in the input image")
    parser.add_argument("--image_dim", type=int, default=512, help="Input image dimension")
    parser.add_argument("--four_channels", type=bool, default=True, help="Whether the gradients should be put as a 4th channel")
    
    args = parser.parse_args()
    
    print("args:", args)
    
    cfg = Config()
    
    cfg.resume = args.cp

    # model
    model = Network(cfg)
    print('=> Load model')

    model.to("cuda:0")
    print('=> Cuda used')
             
    model.load_checkpoint()
             
    print("Model successfully loaded")
    
    original_datapath = os.path.join("data", args.dn)
    
    train_ds = torch.utils.data.DataLoader(create_dataset_generator(original_datapath, "train", data_percentage=1.0), shuffle=False, batch_size=args.bs)
    val_ds = torch.utils.data.DataLoader(create_dataset_generator(original_datapath, "val", data_percentage=1.0), shuffle=False, batch_size=args.bs)
    test_ds = torch.utils.data.DataLoader(create_dataset_generator(original_datapath, "test", data_percentage=1.0), shuffle=False, batch_size=args.bs)
    
    
    if args.four_channels:
        new_datapath = os.path.join("data", "data_edge_model_four_channels" + "_" + args.dn)
        if os.path.exists(new_datapath):
            shutil.rmtree(new_datapath)
    else:
        new_datapath = os.path.join("data", args.model_folder + "_" + args.dn)
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
                dilation_grads = np.clip(gradients + 0.3, 0.0, 1.0)
                gradient_masks = np.clip(np.expand_dims(dilated_masks * dilation_grads, axis=-1), 0.15, 1.0)
                gradient_imgs = np.uint8(orig_imgs * gradient_masks)

                for it, im in enumerate(gradient_imgs):
                    plt.imsave(img_dir + "/" + names[it] + ".png", im)
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

                predictions = torch.sigmoid(model(imgs)[-1]).cpu().detach()
                
                predictions = predictions.numpy()

                predictions = np.transpose(predictions, (0, 2, 3, 1))
                
                predictions[predictions <= 0.5] = 0.5
                
                new = (orig_imgs/255) * predictions 
                
                #new = np.concatenate((orig_imgs/255, predictions), axis=-1)
                for it, n in enumerate(new):
                    np.save(img_dir + "/" + names[it] + ".npy", np.float16(n))
                
