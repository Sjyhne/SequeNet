import tensorflow as tf

from generator import ImageDataset

import os

def create_dataset_generator(datapath, datatype, batch_size=16, image_size=(512, 512), data_percentage=1.0):
    
    heigth, width = image_size

    data_dir = os.path.join(datapath, datatype)

    img_paths = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]

    image_dataset = ImageDataset(img_paths, batch_size, (heigth, width), data_percentage)
    
    return image_dataset

if __name__ == "__main__":

    dpath = os.path.join("data/large_building_area/img_dir")
    dtype = "train"
    data_generator = create_dataset_generator(dpath, dtype, batch_size=8, image_size=(512, 512))

    print(data_generator[0])