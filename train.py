# train.py

from models import MaskRCNN, Config
from config import SimpleConfig
from generator import create_dataset_generator

import tensorflow as tf

tf.compat.v1.disable_eager_execution()

import os

# TODO: Create dataset using the dataset in the maskrcnn utils file ---->  Atleast for the maskrcnn training, maybe use it for everything else aswell

if __name__ == "__main__":

    config = SimpleConfig()
    
    print(config.IMAGES_PER_GPU)

    dpath = "data/large_building_area/img_dir"

    train_ds = create_dataset_generator(dpath, "train", config.IMAGES_PER_GPU, (config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM))
    val_ds = create_dataset_generator(dpath, "val", config.IMAGES_PER_GPU, (config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM))

    model = MaskRCNN("training", config, "./logs")

    model.train(train_ds, val_ds, 1e-4, 10, "heads")