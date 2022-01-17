import tensorflow as tf

from generator import ImageDataset

import os

def create_dataset_generator(datapath, datatype, batch_size=16, image_size=(512, 512)):
    
    heigth, width = image_size

    data_dir = os.path.join(datapath, datatype)

    img_paths = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]

    image_dataset = ImageDataset(img_paths, batch_size, (heigth, width))

    image_dataset_generator = lambda: (i for i in image_dataset)

    dataset = tf.data.Dataset.from_generator(
        image_dataset_generator,
        (tf.uint8, tf.uint8),
        (tf.TensorShape([batch_size, heigth, width, 3]), tf.TensorShape([batch_size, heigth, width, 1]))
    )
    
    return dataset, len(image_dataset)

if __name__ == "__main__":

    dpath = os.path.join("data/large_building_area/img_dir")
    dtype = "train"
    data_generator = create_dataset_generator(dpath, dtype, batch_size=8, image_size=(512, 512))

    print(data_generator.take(6))