# Segmentation Mask Post-Processing
Segmentation mask post-processing using deep neural networks.

Todo when pull

Create dirs:
* logs
* model_output
* output_images


## GOLDMINE
https://github.com/divamgupta/image-segmentation-keras


## Example Scripts

#### Basic training script

> python train.py --epochs 10 --init_lr 1e-3 --image_dim 512 --num_channels 3 --model_type deeplab --batch_size 8 --data_path data/large_building_area/img_dir --data_percentage 0.1 --dataset lba --extra_model True --extra_model_type fcn_8


## Tested models (Meaning they run):

* fcn_8
    * fcn_8_vgg
    * fcn_8_resnet50
    * fcn_8_mobilenet
* fcn_32
    * fcn_32_vgg
    * fcn_32_resnet50
    * fcn_32_mobilenet
* segnet
    * vgg_segnet
    * resnet50_segnet
    * mobilenet_segnet
* unet
    * unet_mini
    * vgg_unet
    * resnet50_unet
    * mobilenet_unet
* deeplab
* hrnet

## Benchmark command:

> python train.py --epochs 50 --init_lr 3e-3 --image_dim 512 --num_channels 3 --model_type deeplab --batch_size 8 --data_path data/large_building_area/img_dir --data_percentage 1.0 --dataset l
ba --main_loss cce

With the following values for each arg

| field | value |
|-------|-------|
|epochs | 50    |
|init_lr| 0.003 |
|image_dim|512|
|num_channels|3|
|num_classes|2|
|model_type|deeplab|
|batch_size|8|
|data_path|"data/large_building_area/img_dir"|
|data_percentage|1|
|loss|"cce"|
|label_smooth|0|
|end_lr|1e-5|
|lr_decay_power|0.5|