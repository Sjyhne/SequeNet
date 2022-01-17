# train.py

from models import finalize_model, build_model
from config import SimpleConfig
from generator import create_dataset_generator

from generator.maskrcnn_dataset import LargeBuildingDataset

import tensorflow as tf

import os

# TODO: Create dataset using the dataset in the maskrcnn utils file ---->  Atleast for the maskrcnn training, maybe use it for everything else aswell

if __name__ == "__main__":
    
    dpath = "data/large_building_area/img_dir/"

    train_ds = create_dataset_generator(dpath, "train", batch_size=2)
    val_ds = create_dataset_generator(dpath, "val", batch_size=1)
    test_ds = create_dataset_generator(dpath, "test", batch_size=1)


    m = finalize_model(build_model(512, 512, 3, 1))

    m.fit(train_ds, epochs=2)