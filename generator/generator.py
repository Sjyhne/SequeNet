import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

from datasets import ImageDataset

import os

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

    img_paths = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]

    image_dataset = ImageDataset(img_paths, batch_size, (heigth, width), data_percentage)
    
    return image_dataset


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

    ds = create_cityscapes_generator("train")

    t = next(iter(ds))

    print(t[0].shape)
    print(t[1].shape)
    
    for i in range(len(t[0])):
        f, x = plt.subplots(1, 2)
        x[0].imshow(t[0][i])
        x[1].imshow(t[1][i])
        plt.show()