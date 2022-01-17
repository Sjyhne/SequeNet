# train.py

from models import finalize_model, build_model
from generator import create_dataset_generator

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import os
#add this for speedy execution (Not eager)
@tf.function
def train_step(m, x, y, loss_func, optimizer):
    with tf.GradientTape() as tape:
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
@tf.function
def evaluate_step(m, x, y, loss_func):
    logits = m(x, training=False)
    loss_val = loss_func(y, logits)
    return loss_val, logits


if __name__ == "__main__":
    
    dpath = "data/large_building_area/img_dir/"

    train_ds, len_train_ds = create_dataset_generator(dpath, "train", batch_size=8)
    val_ds, len_val_ds = create_dataset_generator(dpath, "val", batch_size=8)
    test_ds, len_test_ds = create_dataset_generator(dpath, "test", batch_size=8)


    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    m = finalize_model(build_model(512, 512, 3, 1), optimizer=optimizer)

    # Prepare the metrics.
    train_loss_metric = tf.keras.metrics.BinaryCrossentropy()
    val_loss_metric = tf.keras.metrics.BinaryCrossentropy()
    #train_iou_metric = tf.keras.metrics.BinaryIoU()
    #val_iou_metric = tf.keras.metrics.BinaryIoU()

    epochs = 1

    for epoch in range(epochs):
        print(f"Starting to train for {epochs} epochs")

        for step, (imgs, anns) in tqdm(enumerate(test_ds), total=len_test_ds):
            loss, logits = train_step(m, imgs, anns, bce, optimizer)
            train_loss_metric.update_state(anns, logits)
            #train_iou_metric.update_state(anns, logits)
            if step == len_test_ds:
                break
        
        for step, (imgs, anns) in enumerate(val_ds):
            loss, logits = evaluate_step(m, imgs, anns, bce)
            val_loss_metric.update_state(anns, logits)
            #val_iou_metric.update_state(anns, logits)
            if step == len_val_ds:
                break


        # Display metrics at the end of each epoch.
        train_loss = train_loss_metric.result()
        val_loss = val_loss_metric.result()
        print("Train loss: %.4f | Val loss: %.4f" % (float(train_loss), float(val_loss)))

        # Reset training metrics at the end of each epoch
        train_loss_metric.reset_states()
        #train_iou_metric.reset_states()
        val_loss_metric.reset_states()
        #val_iou_metric.reset_states()
    
    for step, (imgs, anns) in enumerate(val_ds):
        logits = m(imgs, training=False)
        sigmoid_logits = tf.math.sigmoid(logits).numpy()
        rounded_sigmoid_logits = tf.math.round(sigmoid_logits).numpy()
        
        for i in range(len(logits)):
            f, x = plt.subplots(1, 3)
            x[0].imshow(sigmoid_logits[i].astype(np.uint8))
            x[1].imshow(rounded_sigmoid_logits[i].astype(np.uint8))
            x[2].imshow(anns[i].numpy().astype(np.uint8))
            plt.savefig(f"{i}_test_pic.png")
        if step == 5:
            break