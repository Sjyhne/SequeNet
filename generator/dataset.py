import torch
import numpy as np
import cv2 as cv

import random
import os

import albumentations as A

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, bsize, img_size, data_percentage=1.0, four_channels=False, transform=False) -> None:
        self.image_paths = image_paths
        random.shuffle(self.image_paths)
        self.image_paths = self.image_paths[:int(len(self.image_paths) * data_percentage)]
        self.label_paths = self.get_label_paths()
        self.img_size = img_size
        self.four_channels = four_channels
        self.transform = transform
        
        #self.image_batches, self.label_batches = self.generate_batches()
    
    def get_label_path(self, path):
        tmp = path.split("/")
        path = os.path.join(tmp[0], "large_building_area", tmp[2], "ann_dir", "/".join(tmp[4:]))
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
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label_path = self.label_paths[idx]
        
        if not self.four_channels:
            img = cv.imread(image_path)
        else:
            img = np.load(image_path)
        
        if not self.four_channels:
            lab = cv.imread(label_path.replace("tiff", "tiff"), cv.IMREAD_GRAYSCALE)
        else:
            lab = cv.imread(label_path.replace("npy", "tiff"), cv.IMREAD_GRAYSCALE)
        #lab = lab.reshape(self.img_size[0], self.img_size[1], 1)
        lab[lab == 30] = 0
        lab[lab == 255] = 1
        #lab = lab.squeeze()
        name = image_path.split("/")[-1].split(".")[0]
        
        
        try:
            dist_map = np.int16(np.load(label_path.split(".")[0].replace("edge_dir", "ann_dir") + ".npy").squeeze())
        except Exception as e:
            #print(e)
            dist_map = []
            
        
        if self.transform:
            aug = A.Compose([
                A.VerticalFlip(p=0.5),              
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.5),
                A.Transpose(p=0.4),
                A.RandomBrightness(always_apply=False, p=0.4, limit=(-0.30, 0.30))]
            )
            
            augmented = aug(image=img, mask=lab)
            
            img = augmented["image"]
            lab = augmented["mask"]
            
        
        if not self.four_channels:
            tensor_img = torch.tensor(img/255, dtype=torch.float32)
        else:
            tensor_img = torch.tensor(img, dtype=torch.float32)
        tensor_lab = torch.tensor(lab, dtype=torch.int64)
        dist_map = torch.tensor(dist_map, dtype=torch.float32)
        
        #img_pad = (0, 0, 6, 6, 6, 6)
        #lab_pad = (6, 6, 6, 6)
        
        
        #tensor_img = torch.nn.functional.pad(tensor_img, img_pad, "constant", 0)
        #tensor_lab = torch.nn.functional.pad(tensor_lab, lab_pad, "constant", 0)
        
        #img = np.pad(img, [(6, 6), (6, 6), (0, 0)], "constant")
        
        """
        for i in range(self.bsize):
            img = cv.imread(image_paths[i], cv.IMREAD_COLOR)
            lab = cv.imread(label_paths[i], cv.IMREAD_GRAYSCALE)
            lab = lab.reshape(self.img_size[0], self.img_size[1], 1)
            if lab[0, 0] > 1:
                lab[lab == 30] = 0
                lab[lab == 215] = 1

            lab = lab.squeeze()
            imgs[i] = img
            labs[i] = lab
            names.append(image_paths[i].split("/")[-1].split(".")[0])
            try:
                dist_maps[i] = np.load(os.path.join("data/large_building_area/ann_dir", "/".join(label_paths[i].split("/")[-2:]).split(".")[0] + ".npz.npy"))
            except Exception as e:
                pass
        tensor_imgs = torch.tensor(imgs/255, dtype=torch.float32)
        tensor_labs = torch.tensor(labs, dtype=torch.int64)
        """
        
        res = {"img": tensor_img, "lab": tensor_lab, "name": name, "dist_map": dist_map, "orig_img": img}
        
        return res

if __name__ == "__main__":
    pass