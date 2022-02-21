import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

from .dataset import ImageDataset

import os
import shutil

split_images_to_half = lambda x: (tf.reshape(
    tf.image.extract_patches(
        images=tf.expand_dims(x["image_left"], 0),
        sizes=[1, 512, 512, 3],
        strides=[1, 512, 512, 3],
        rates=[1, 1, 1, 1],
        padding='VALID'), (8, 512, 512, 3)), 
    tf.reshape(tf.image.extract_patches(
        images=tf.expand_dims(x["segmentation_label"], 0),
        sizes=[1, 512, 512, 1],
        strides=[1, 512, 512, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'), (8, 512, 512, 1))
)

def create_dataset_generator(datapath, datatype, batch_size=16, image_size=(512, 512), data_percentage=1.0, create_dist=False):
    
    heigth, width = image_size

    data_dir = os.path.join(datapath, datatype)

    img_paths = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
    
    if create_dist:
        print(os.path.join("/".join(datapath.split("/")[:-1]), "ann_dir", datatype))
        generate_dist_maps(os.path.join("/".join(datapath.split("/")[:-1]), "ann_dir", datatype))

    image_dataset = ImageDataset(img_paths, batch_size, (heigth, width), data_percentage)
    
    return image_dataset

from keras import backend as K
import numpy as np
import tensorflow as tf
from scipy.ndimage import distance_transform_edt as distance
import cv2 as cv
from operator import itemgetter
from tqdm import tqdm
import os
import json

# Tools
def kl_div(a,b): # q,p
    return tf.nn.softmax(b, axis=1) * (tf.nn.log_softmax(b, axis=1) - tf.nn.log_softmax(a, axis=1))   

def one_hot2dist(seg):
    res = np.zeros_like(seg)
    for i in range(len(seg)):
        posmask = seg[i].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            res[i] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    return res

def class2one_hot(seg, C):
    seg = tf.expand_dims(seg, axis=0) if len(seg.shape) == 2 else seg
    res = tf.cast(tf.stack([seg == c for c in range(C)], axis=1), tf.int32)
    return res

def dist_map_transform(value):
    value = tf.expand_dims(value, axis=0)
    value = tf.cast(value, dtype=tf.int64)
    value = class2one_hot(value, C=1)
    getter = itemgetter(0)
    value = getter(value)
    #value = value.cpu().numpy()
    value = value.numpy()
    value = one_hot2dist(value)
    return tf.convert_to_tensor(value, dtype=tf.float32)

def calc_dist_map(seg):
    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
       
        res = res / np.amax(res)
    
    return res

def get_abl_dist_maps(target):
    #target_detach = target.clone().detach()
    target_detach = target
    dist_maps = tf.concat([dist_map_transform(target_detach[i]) for i in range(target_detach.shape[0])], axis=0)
    out = -dist_maps
    out = tf.where(out>0, out, tf.zeros_like(out))

    return out

def gt2boundary(gt, ignore_label=-1):  # gt NHW
    gt_lr = gt[:,1:,:]-gt[:,:-1,:]  # NHW
    gt_ud = gt[:,:,1:]-gt[:,:,:-1]

    gt_lr = tf.cast(gt_lr, dtype=tf.int16)
    gt_ud = tf.cast(gt_ud, dtype=tf.int16)
    gt_lr = tf.pad(gt_lr, [[0,0],[0,1],[0,0]], mode='CONSTANT', constant_values=0) != 0 
    gt_ud = tf.pad(gt_ud, [[0,0],[0,0],[0,1]], mode='CONSTANT', constant_values=0) != 0
    gt_lr = tf.cast(gt_lr, dtype=tf.int16)
    gt_ud = tf.cast(gt_ud, dtype=tf.int16)

    gt_combine = gt_lr+gt_ud
    del gt_lr
    del gt_ud

    gt = tf.cast(gt, dtype=tf.int16)

    # set 'ignore area' to all boundary
    gt_combine += tf.cast((gt==ignore_label), dtype=tf.int16)

    return gt_combine > 0

def calc_abl_dist_map(seg):
    
    if len(seg.shape) == 4:
        target = tf.math.argmax(seg, axis=-1)

    gt_boundary = gt2boundary(seg, ignore_label=255)
    
    #dist_maps = self.get_dist_maps(gt_boundary).cuda() # <-- it will slow down the training, you can put it to dataloader.
    dist_maps = get_abl_dist_maps(gt_boundary)
    
    return dist_maps
    # TODO: Continue here by implementing the dist-generation
        

def calc_dist_map_batch(y_true):
    y_true_numpy = y_true.numpy()
    return_array = np.array([calc_dist_map(y) for y in y_true_numpy]).astype(np.float32)
    normalized_return_array = return_array / np.amax(return_array)

def generate_dist_maps(mask_folder):
    dist_maps = {}
    
    for _, file in tqdm(enumerate(os.listdir(mask_folder)), total=len(os.listdir(mask_folder))):
        if file.split(".")[-1] == "tiff":
            lab = cv.imread(os.path.join(mask_folder, file), cv.IMREAD_GRAYSCALE)
            lab = lab.reshape(lab.shape[0], lab.shape[1], 1)
            if lab[0, 0] > 1:
                lab[lab == 30] = 0
                lab[lab == 215] = 1
            #lab = tf.keras.utils.to_categorical(lab, num_classes=2)
            dist_map = calc_abl_dist_map(lab)[0]
            np.save(os.path.join(mask_folder, file.split(".")[0] + ".npz"), dist_map)
            
def create_dataset_from_model(dataset, dataset_type, args):
    base_path = os.path.join("data", args.load_model + "_large_building_area")
    
    print(args)
    
    if os.path.exists(base_path) and args.overwrite_dataset == False:
        print("Data already exists, and overwrite is False")
        
        return create_dataset_generator(os.path.join(base_path, "img_dir"), dataset_type, args.batch_size, data_percentage=args.data_percentage)
    
    print("Please dont go here")
    exit()
    
    model_path = os.path.join("model_output", args.load_model)
    for file in os.listdir(model_path):
        if "best_model" in file:
            model_path = os.path.join("model_output", args.load_model, file)
    print("Found this model:", model_path)
    try:
        model = tf.keras.models.load_model(model_path)
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
    
    for step, (imgs, anns, names) in tqdm(enumerate(dataset), total=len(dataset)):
        preds = model(imgs)
        anns = tf.math.argmax(anns, axis=-1)
        masks = tf.math.argmax(preds, axis=-1)
        grads = preds[:, :, :, 1]
        logits = (tf.cast(imgs, dtype=tf.float32) * tf.expand_dims(tf.clip_by_value(preds[:, :, :, 1], 0.3, 1.0), axis=-1))/255
        for i, logit in enumerate(logits):
            plt.imsave(os.path.join(base_path, "img_dir", dataset_type, names[i] + ".png"), logit.numpy())
            plt.imsave(os.path.join(base_path, "ann_dir", dataset_type, names[i] + ".png"), anns[i].numpy())
            plt.imsave(os.path.join(base_path, "mask_dir", dataset_type, names[i] + ".png"), masks[i].numpy())
            plt.imsave(os.path.join(base_path, "grad_dir", dataset_type, names[i] + ".png"), grads[i].numpy())
            
    
    return create_dataset_generator(os.path.join(base_path, "img_dir"), dataset_type, args.batch_size, data_percentage=args.data_percentage)


def patchify_images(tupl):

    img = tupl["image_left"]
    ann = tupl["segmentation_label"]

    image_patches, annotation_patches = [], []

    for y in range(int(2)):
        for x in range(int(4)):
            tmp_img = img[0, y * 512 : (y+1) * 512, x * 512 : (x+1) * 512, :]
            tmp_ann = ann[0, y * 512 : (y+1) * 512, x * 512 : (x+1) * 512, :]
            image_patches.append(tmp_img)
            annotation_patches.append(tmp_ann)
    
    return tf.convert_to_tensor(image_patches, dtype=tf.uint8), tf.convert_to_tensor(annotation_patches, dtype=tf.uint8)

def create_cityscapes_generator(datatype):
    ds = tfds.load("cityscapes", split=datatype, shuffle_files=True, batch_size=1)
    ds = ds.map(patchify_images)
    return ds

if __name__ == "__main__":

    ds = create_dataset_generator("data/large_building_area/img_dir", "val")

    for img, ann in ds:
        print(ann.shape)
    