import matplotlib.pyplot as plt
from models import Dataset
from models.maskrcnn_utils import extract_bboxes

from pycocotools import mask as maskUtils

import numpy as np
import tensorflow as tf

import os

import skimage.io

class LargeBuildingDataset(Dataset):
    
    def load_dataset(self, dataset_dir, dataset_type, is_train=True):
        self.add_class("dataset", 1, "building")

        images_dir = os.path.join(dataset_dir, "img_dir", dataset_type)
        annotations_dir = os.path.join(dataset_dir, "ann_dir", dataset_type)


        for filename in os.listdir(images_dir):
            image_id = filename[:-5]

            img_path = os.path.join(images_dir, filename)
            ann_path = os.path.join(annotations_dir, filename)

            self.add_image("dataset", image_id=image_id, path=img_path, annotation=ann_path)
        
    def load_mask(self, image_id):
        info = self.image_info[image_id]
        path = info["annotation"]

        m = skimage.io.imread(path).astype(np.uint8).reshape(512, 512, 1)

        print(m.max(), m.min())

        print(self.annToMask(m, 512, 512))

        print(extract_bboxes(m))

        return m, np.ones([m.shape[-1]], dtype=np.int32)

    def annToRLE(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        segm = ann
        if isinstance(segm, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, height, width)
            rle = maskUtils.merge(rles)
        elif isinstance(segm['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            # rle
            rle = ann   
        return rle

    def annToMask(self, ann, height, width):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann, height, width)
        m = maskUtils.decode(rle)
        return m

if __name__ == "__main__":
    dataset = LargeBuildingDataset()

    dataset.load_dataset("data/large_building_area", "val", True)
    
    m, l = dataset.load_mask(0)
    r = dataset.load_image(0)
    print(m.shape)
    print(l)
    print(r.shape)

    f, x = plt.subplots(1, 2)
    x[0].imshow(m)
    x[1].imshow(r)
    plt.show()