# train.py

from models import finalize_model, build_model
from config import SimpleConfig
from generator import create_dataset_generator

from generator.maskrcnn_dataset import LargeBuildingDataset

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import os

# TODO: Create dataset using the dataset in the maskrcnn utils file ---->  Atleast for the maskrcnn training, maybe use it for everything else aswell

if __name__ == "__main__":
    
    dpath = "data/large_building_area/img_dir/"

    train_ds = create_dataset_generator(dpath, "train", batch_size=2)
    val_ds = create_dataset_generator(dpath, "val", batch_size=1)
    test_ds = create_dataset_generator(dpath, "test", batch_size=1)


    optimizer = tf.keras.optimizers.Adam()

    m = finalize_model(build_model(512, 512, 3, 1), optimizer=optimizer)

    epochs = 2

    for epoch in range(epochs):
        print(f"Starting to train for {epochs} epochs")

        for step, (imgs, anns) in enumerate(train_ds):
            print("imgs dim:", imgs.shape, "| anns dim:", anns.shape)
            with tf.GradientTape() as tape:

                logits = m(imgs, training=True)

                sigmoid_logits = tf.math.sigmoid(logits)

                bce = tf.keras.losses.BinaryCrossentropy(
                    from_logits=False
                )

                bce_loss = bce(anns, sigmoid_logits)

                print("bce_loss:", bce_loss)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(bce_loss, m.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, m.trainable_weights))


