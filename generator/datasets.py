import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

import random
import os

class ImageDataset:
    def __init__(self, image_paths, bsize, img_size, data_percentage=1.0) -> None:
        self.image_paths = image_paths
        random.shuffle(self.image_paths)
        self.image_paths = self.image_paths[:int(len(self.image_paths) * data_percentage)]
        self.label_paths = self.get_label_paths()
        self.bsize = bsize
        self.img_size = img_size

        self.image_batches, self.label_batches = self.generate_batches()
    
    def get_label_path(self, path):
        tmp = path.split("/")
        path = os.path.join("/".join(tmp[:2]), "ann_dir", "/".join(tmp[3:]))
        return path
    
    def get_label_paths(self):
        label_paths = []
        for path in self.image_paths:
            path = self.get_label_path(path)
            label_paths.append(path)
        
        assert len(label_paths) == len(self.image_paths)

        return label_paths
    
    def generate_batches(self):
        image_batch_paths = []
        label_batch_paths = []
        tmp_image_batch = []
        tmp_label_batch = []
        for i in range(len(self.image_paths)):
            tmp_image_batch.append(self.image_paths[i])
            tmp_label_batch.append(self.label_paths[i])
            if len(tmp_image_batch) == self.bsize:
                image_batch_paths.append(tmp_image_batch)
                label_batch_paths.append(tmp_label_batch)
                tmp_image_batch = []
                tmp_label_batch = []
        
        if len(tmp_image_batch) != 0:
            for i in range(self.bsize - len(tmp_image_batch)):
                img_path = random.choice(self.image_paths)
                lab_path = self.get_label_path(img_path)
                tmp_image_batch.append(img_path)
                tmp_label_batch.append(lab_path)
            
            image_batch_paths.append(tmp_image_batch)
            label_batch_paths.append(tmp_label_batch)
        
        return image_batch_paths, label_batch_paths

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.bsize))
    
    def __getitem__(self, idx):
        image_paths = self.image_batches[idx]
        label_paths = self.label_batches[idx]
        imgs = np.ndarray((self.bsize, self.img_size[0], self.img_size[1], 3))
        labs = np.ndarray((self.bsize, self.img_size[0], self.img_size[1], 1))
        for i in range(self.bsize):
            img = cv.imread(image_paths[i], cv.IMREAD_COLOR)
            lab = cv.imread(label_paths[i], cv.IMREAD_GRAYSCALE).reshape(self.img_size[0], self.img_size[1], 1)
            imgs[i] = img
            labs[i] = lab
        tensor_imgs = tf.convert_to_tensor(imgs, dtype=tf.int64)
        tensor_labs = tf.convert_to_tensor(labs, dtype=tf.uint8)
        
        return tensor_imgs, tensor_labs
