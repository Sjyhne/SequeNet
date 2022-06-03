import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import torch

from tqdm import tqdm

from .dataset import ImageDataset

import os
import shutil

class ModelClass(torch.nn.Module):
    def __init__(self):
        print("Replace me with the correct model class!")

def create_dataset_generator(datapath, datatype, batch_size=16, image_size=(512, 512), data_percentage=1.0, create_dist=False, four_channels=False, transform=False):
    
    heigth, width = image_size

    data_dir = os.path.join(datapath, datatype, "img_dir")
    
    print(data_dir)

    img_paths = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.npy', ".tif"))]
    
    print(len(img_paths))
    
    if create_dist:
        print(os.path.join("/".join(datapath.split("/")[:-1]), "ann_dir", datatype))
        generate_dist_maps(os.path.join("/".join(datapath.split("/")[:-1]), "ann_dir", datatype))

    image_dataset = ImageDataset(img_paths, batch_size, (heigth, width), data_percentage, four_channels=four_channels, transform=transform)
    
    return image_dataset


def calc_abl_dist_map(seg):

    # TODO: get the definition from the github repo
    
    pass

def generate_dist_maps(mask_folder):
    dist_maps = {}
    
    for _, file in tqdm(enumerate(os.listdir(mask_folder)), total=len(os.listdir(mask_folder))):
        if file.split(".")[-1] == "tiff":
            lab = cv.imread(os.path.join(mask_folder, file), cv.IMREAD_GRAYSCALE)
            lab = lab.reshape(lab.shape[0], lab.shape[1], 1)
            if lab[0, 0] > 1:
                lab[lab == 30] = 0
                lab[lab == 215] = 1
            dist_map = calc_abl_dist_map(lab)[0]
            np.save(os.path.join(mask_folder, file.split(".")[0] + ".npz"), dist_map)

def cannify_images(imgs, thresholds):
    cannified = np.ndarray((imgs.shape[0], imgs.shape[1], imgs.shape[2], 1))
    imgs = np.uint8(imgs.numpy())
    
    for i, img in enumerate(imgs):
        cannified[i] = np.expand_dims(cv.Canny(img, thresholds[0], thresholds[1]), axis=-1)
        
    return cannified

def lsdify_images(imgs, lsd):
    lines = []
    imgs = np.uint8(imgs.numpy())
    for img in imgs:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        line = lsd.detect(img)[0]
        lines.append(line)
    
    return lines

def dilate_masks(masks, kernel):
    dilated = np.ndarray((masks.shape))
    masks = np.uint8(masks.numpy())
    for i, mask in enumerate(masks):
        dilated[i] = cv.dilate(mask, np.ones(kernel))
    
    return dilated
            
def create_dataset_from_model(dataset, dataset_type, args):
    base_path = os.path.join("data", args.load_model + "_large_building_area")
    
    print(args)
    
    if os.path.exists(base_path) and args.overwrite_dataset == False:
        print("Data already exists, and overwrite is False")
        
        return create_dataset_generator(os.path.join(base_path, "img_dir"), dataset_type, args.batch_size, data_percentage=args.data_percentage)
    
    model_path = os.path.join("model_output", args.load_model)
    for file in os.listdir(model_path):
        if "best_model" in file:
            model_path = os.path.join("model_output", args.load_model, file)
    print("Found this model:", model_path)
    try:
        # TODO: Replace with torch load model
        model = ModelClass()
        model = model.load_state_dict(torch.load(model_path))
        print("Successfully loaded model", args.load_model, "from", model_path)
    except Exception as e:
        print("Something went wrong loading", model_path, "|", args.load_model)
        print(e)
        exit()

        # Create folder for data if not exist, delete if already exist
    if os.path.exists(os.path.join(base_path, "img_dir", dataset_type)):
        shutil.rmtree(os.path.join(base_path, "img_dir", dataset_type))
        shutil.rmtree(os.path.join(base_path, "ann_dir", dataset_type))
        shutil.rmtree(os.path.join(base_path, "mask_dir", dataset_type))
        shutil.rmtree(os.path.join(base_path, "grad_dir", dataset_type))
        
        
    
    os.makedirs(os.path.join(base_path, "img_dir", dataset_type))
    os.makedirs(os.path.join(base_path, "ann_dir", dataset_type))
    os.makedirs(os.path.join(base_path, "mask_dir", dataset_type))
    os.makedirs(os.path.join(base_path, "grad_dir", dataset_type))
    
    lsd = cv.createLineSegmentDetector(0, ang_th=40.0)
    
    for step, (imgs, anns, names, _) in tqdm(enumerate(dataset), total=len(dataset)):
        preds = model(imgs)
        # TODO: Replace argmax with torch argmax
        anns = torch.argmax(anns, axis=-1)
        masks = torch.argmax(torch.nn.softmax(preds, axis=-1), axis=-1)
        # TODO: Describe the constants with some value
        grads = np.clip(torch.nn.softmax(preds, axis=-1)[:, :, :, 1].numpy() + 0.5, 0.2, 1.0)
        #canny = cannify_images(imgs, (120, 200))
        lines = lsdify_images(imgs, lsd)
        dilated_masks = np.clip(dilate_masks(masks, (30, 30)), 0.15, 1.0)
        build_grad = np.expand_dims(grads * dilated_masks, axis=-1)
        #canny = build_grad * canny
        imgs = np.uint8(imgs.numpy())
        new_imgs = np.ndarray(imgs.shape)
        for i, img in enumerate(imgs):
            new_imgs[i] = lsd.drawSegments(img, lines[i])
        logits = np.clip(new_imgs * build_grad, 0, 255)
        logits = np.uint8(logits)
        for i, logit in enumerate(logits):
            plt.imsave(os.path.join(base_path, "img_dir", dataset_type, names[i] + ".png"), logit)
            plt.imsave(os.path.join(base_path, "ann_dir", dataset_type, names[i] + ".png"), anns[i].numpy())
            plt.imsave(os.path.join(base_path, "mask_dir", dataset_type, names[i] + ".png"), dilated_masks[i])
            plt.imsave(os.path.join(base_path, "grad_dir", dataset_type, names[i] + ".png"), np.squeeze(build_grad)[i])
        
    return create_dataset_generator(os.path.join(base_path, "img_dir"), dataset_type, args.batch_size, data_percentage=args.data_percentage)

    
if __name__ == "__main__":

    ds = create_dataset_generator("data/large_building_area/img_dir", "val")

    for img, ann in ds:
        print(ann.shape)
    