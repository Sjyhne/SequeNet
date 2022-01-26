# train.py

from models import finalize_model, build_model, get_model, DeeplabV3Plus
from models.metrics import mean_iou
from generator import create_dataset_generator, create_cityscapes_generator
from train_utils import calc_biou, remove_all_folders_in_path, store_images
from train_utils import display_and_store_metrics, save_best_model, calculate_sample_weight

import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import tensorflow_datasets as tfds

import argparse
import os

NUM_CLASSES = 2

def train_step(m, x, y, loss_func, optimizer):
    with tf.GradientTape()  as tape:
        logits = m(x, training=True)
        loss_val = loss_func(y, logits)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss_val, m.trainable_variables)
    
    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, m.trainable_variables))
    return loss_val, tf.nn.softmax(logits, axis=-1)

def evaluate_step(m, x, y, loss_func):
    logits = m(x, training=False)
    loss_val = loss_func(y, logits)
    return loss_val, logits

def train(args, train_ds, val_ds):
    
    # Add learning rate scheduler to the optimizer -- Believe that should work -- CosineWarmStart or something
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.init_lr)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    if args.model_type == "unet":
        m = get_model((512, 512), args.num_classes)
    elif args.model_type == "deeplab":
        m = DeeplabV3Plus(512, args.num_classes)
    else:
        print("Not a supported model-type")
        exit()
        
    # Prepare the metrics.
    train_loss_metric = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)
    val_loss_metric = tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True)
    train_miou_metric = []
    val_miou_metric = []
    train_biou_metric = []
    val_biou_metric = []
    
    remove_all_folders_in_path("output_images/train_ds/")
    remove_all_folders_in_path("output_images/val_ds/")

    epochs = args.epochs
    
    best_loss_value = 10_000
    
    print(f"Starting to train for {epochs} epochs")
    for epoch in range(epochs):
        print(f"Epoch: {epoch}/{epochs}")
        print()
        for step, (imgs, anns) in tqdm(enumerate(train_ds), total=len(train_ds)):
            loss, logits = train_step(m, imgs, anns, loss_fn, optimizer)
            pred_images = tf.math.argmax(tf.nn.softmax(logits, axis=-1), axis=-1)
            iou_anns = tf.squeeze(anns, axis=-1)
            miou = mean_iou(iou_anns, pred_images).numpy()
            biou = calc_biou(pred_images, iou_anns)
            
            train_loss_metric.update_state(anns, logits)
            train_miou_metric.append(miou)
            train_biou_metric.append(biou)
            if step == len(train_ds) - 1:
                store_images(pred_images, iou_anns, imgs, f"output_images/train_ds/{epoch}")
        
        for step, (imgs, anns) in enumerate(val_ds):
            loss, logits = evaluate_step(m, imgs, anns, loss_fn)
            pred_images = tf.math.argmax(tf.nn.softmax(logits, axis=-1), axis=-1)
            iou_anns = tf.squeeze(anns, axis=-1)
            miou = mean_iou(iou_anns, pred_images).numpy()
            biou = calc_biou(pred_images, iou_anns)
            
            val_loss_metric.update_state(anns, logits)
            val_miou_metric.append(miou)
            val_biou_metric.append(biou)
            if step == len(val_ds) - 1:
                store_images(pred_images, iou_anns, imgs, f"output_images/val_ds/{epoch}")

        display_and_store_metrics(
            train_loss_metric, val_loss_metric,
            train_miou_metric, val_miou_metric,
            train_biou_metric, val_biou_metric
        )
        
        best_loss_value = save_best_model(m, val_loss_metric.result(), best_loss_value, epoch)

        # Reset training metrics at the end of each epoch
        train_loss_metric.reset_states()
        val_loss_metric.reset_states()
        train_miou_metric = []
        val_miou_metric = []
        train_biou_metric = []
        val_biou_metric = []


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Add custom arguments for the training of the model(s)")

    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--init_lr", type=float, default=1e-3, help="The initial learning rate")
    parser.add_argument("--image_dim", type=tuple, default=(512, 512), help="The dimensions of the input image")
    parser.add_argument("--num_channels", type=int, default=3, help="Number of channels in input image")
    parser.add_argument("--num_classes", type=int, default=2, help="The number of classes to predict")
    parser.add_argument("--model_type", type=str, default="unet", help="The model type to be trained", choices=["unet", "deeplab"])
    parser.add_argument("--batch_size", type=int, default=8, help="The batchsize used for the training")
    parser.add_argument("--data_path", type=str, default="data/large_building_area/img_dir", help="Path to data used for training")
    parser.add_argument("--data_percentage", type=float, default=1.0, help="The percentage size of data to be used during training")

    args = parser.parse_args()


    if args.dataset == "lba":
        train_ds = create_dataset_generator(args.data_path, "train", batch_size=args.batch_size, data_percentage=args.data_percentage)
        val_ds = create_dataset_generator(args.data_path, "val", batch_size=args.batch_size, data_percentage=args.data_percentage)
        test_ds = create_dataset_generator(args.data_path, "test", batch_sizsde=args.batch_size, data_percentage=args.data_percentage)
    elif args.dataset == "cityscapes":
        train_ds = create_cityscapes_generator("train")
        val_ds = create_cityscapes_generator("val")
        test_ds = create_cityscapes_generator("test")

    train(args, train_ds, val_ds)