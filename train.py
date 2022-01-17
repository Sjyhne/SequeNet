# train.py

from models import finalize_model, build_model
from config import SimpleConfig
from generator import create_dataset_generator

from generator.maskrcnn_dataset import LargeBuildingDataset

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import os
#add this for speedy execution (Not eager)
#@tf.function

def train_step(m, x, y, loss_func, optimizer):
    with tf.GradientTape as tape:
        logits = m(x, training=True)
        loss_val = loss_func(y, logits)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss_val, m.trainable_weights)
    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, m.trainable_weights))
    return loss_val, logits

#add this for speedy execution (Not eager)
#@tf.function
def evaluate_step(m, x, y, loss_func):
    logits = m(x, training=False)
    loss_val = loss_func(y, logits)
    return loss_val, logits


if __name__ == "__main__":
    
    dpath = "data/large_building_area/img_dir/"

    train_ds = create_dataset_generator(dpath, "train", batch_size=2)
    val_ds = create_dataset_generator(dpath, "val", batch_size=1)
    test_ds = create_dataset_generator(dpath, "test", batch_size=1)


    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    bce = tf.keras.losses.BinaryCrossentropy(logits=True)

    m = finalize_model(build_model(512, 512, 3, 1), optimizer=optimizer)

    # Prepare the metrics.
    train_loss_metric = tf.keras.metrics.BinaryCrossentropy()
    val_loss_metric = tf.keras.metrics.BinaryCrossentropy()
    train_iou_metric = tf.keras.metrics.BinaryIoU()
    val_iou_metric = tf.keras.metrics.BinaryIoU()

    epochs = 2

    for epoch in range(epochs):
        print(f"Starting to train for {epochs} epochs")

        for step, (imgs, anns) in enumerate(train_ds):
            loss, logits = train_step(m, imgs, anns, bce, optimizer)
            train_loss_metric.update_state(anns, logits)
            train_iou_metric.update_state(anns, logits)
        
        for imgs, anns in val_ds:
            loss, logits = evaluate_step(m, imgs, anns, bce)
            val_loss_metric.update_state(anns, logits)
            val_iou_metric.update_state(anns, logits)


        # Display metrics at the end of each epoch.
        train_loss = train_loss_metric.result()
        print("Train loss: %.4f | Train IoU: %.4f | Val loss: %.4f | Val IoU: %.4f" % (float(train_loss),))

        # Reset training metrics at the end of each epoch
        train_loss_metric.reset_states()
        train_iou_metric.reset_states()
        val_loss_metric.reset_states()
        val_iou_metric.reset_states()