import tensorflow as tf
import numpy as np
import cv2 as cv

import random
import os

class ImageDataset:
    def __init__(self, image_paths, bsize, img_size) -> None:
        self.image_paths = image_paths
        random.shuffle(self.image_paths)
        self.label_paths = self.get_label_paths()
        self.bsize = bsize
        self.img_size = img_size

        self.image_ids = range(len(self.image_paths))

        self.image_batches, self.label_batches = self.generate_batches()
    
    def get_label_paths(self):
        label_paths = []
        for path in self.image_paths:
            tmp = path.split("/")
            path = os.path.join("/".join(tmp[:2]), "ann_dir", "/".join(tmp[3:]))
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
                tmp_image_batch.append(random.choice(self.image_paths))
                tmp_label_batch.append(random.choice(self.label_paths))
            
            image_batch_paths.append(tmp_image_batch)
            label_batch_paths.append(tmp_label_batch)
        
        return image_batch_paths, label_batch_paths

    def __len__(self):
        return np.ceil(len(self.image_paths) / self.bsize)
    
    def __getitem__(self, idx):
        image_paths = self.image_batches[idx]
        label_paths = self.label_batches[idx]
        imgs = np.ndarray((self.bsize, self.img_size[0], self.img_size[1], 3))
        labs = np.ndarray((self.bsize, self.img_size[0], self.img_size[1], 1))
        for i in range(self.bsize):
            img = cv.imread(image_paths[i], cv.IMREAD_COLOR)
            lab = cv.imread(label_paths[i], cv.IMREAD_GRAYSCALE)
            imgs[i] = img
            labs[i] = lab.reshape(self.img_size[0], self.img_size[1], 1)
        
        tensor_imgs = tf.convert_to_tensor(imgs, dtype=tf.uint8)
        tensor_labs = tf.convert_to_tensor(labs, dtype=tf.uint8)

        return tensor_imgs, tensor_labs
