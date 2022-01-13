import tensorflow as tf

from datasets import ImageDataset

import os

def create_dataset_generator(datapath, datatype, batch_size=16, image_size=(512, 512)):
    
    heigth, width = image_size

    data_dir = os.path.join(datapath, datatype)

    img_paths = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]

    image_dataset = ImageDataset(img_paths, batch_size, (heigth, width))

    image_dataset_generator = lambda: (i for i in image_dataset)

    dataset = tf.data.Dataset.from_generator(
        image_dataset_generator, 
        output_signature=(
            tf.TensorSpec(shape=(batch_size, heigth, width, 3), dtype=tf.uint8),
            tf.TensorSpec(shape=(batch_size, heigth, width), dtype=tf.uint8)
        ))
    
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

if __name__ == "__main__":

    dpath = os.path.join("data/large_building_area/img_dir")
    dtype = "train"
    data_generator = create_dataset_generator(dpath, dtype, batch_size=8, image_size=(512, 512))

    print(data_generator.take(6))