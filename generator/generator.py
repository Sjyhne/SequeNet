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

def create_dataset_generator(datapath, datatype, batch_size=16, image_size=(512, 512), data_percentage=1.0):
    
    heigth, width = image_size

    data_dir = os.path.join(datapath, datatype)

    img_paths = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]

    image_dataset = ImageDataset(img_paths, batch_size, (heigth, width), data_percentage)
    
    return image_dataset

def create_dataset_from_model(model, dataset, dataset_type, args):
    base_path = os.path.join("data", args.load_model + "_large_building_area")
    
    if os.path.exists(base_path) and args.overwrite_dataset == False:
        print("Data already exists, and overwrite is False")
        
        return create_dataset_generator(os.path.join(base_path, "img_dir"), dataset_type, args.batch_size, data_percentage=args.data_percentage)
    
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
    