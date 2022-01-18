# train.py

from models import finalize_model, build_model
from models.metrics import mean_iou, boundary_iou
from generator import create_dataset_generator

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import os
import shutil

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
    
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0001)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

    m = finalize_model(build_model(512, 512, 3, 2), optimizer=optimizer)

    # Prepare the metrics.
    train_loss_metric = tf.keras.metrics.CategoricalCrossentropy(from_logits=True)
    val_loss_metric = tf.keras.metrics.CategoricalCrossentropy(from_logits=True)
    train_miou_metric = []
    val_miou_metric = []
    train_biou_metric = []
    val_biou_metric = []
    
    for folder in os.listdir("output_images/train_ds/"):
        shutil.rmtree("output_images/train_ds/" + folder)
    
    for folder in os.listdir("output_images/val_ds/"):
        shutil.rmtree("output_images/val_ds/" + folder)

    epochs = 10
    
    print(f"Starting to train for {epochs} epochs")
    for epoch in range(epochs):
        print(f"Epoch: {epoch}/{epochs}")
        print()
        for step, (imgs, anns) in tqdm(enumerate(train_ds), total=len_train_ds):
            loss, logits = train_step(m, imgs, anns, loss_fn, optimizer)
            pred_images = tf.squeeze(tf.math.argmax(tf.nn.softmax(logits, axis=-1), axis=-1))
            iou_anns = tf.squeeze(tf.math.argmax(anns, axis=-1))
            miou = mean_iou(iou_anns, pred_images).numpy()
            biou = []
            for i, pi in enumerate(pred_images):
                biou.append(boundary_iou(iou_anns[i].numpy().astype("uint8"), pi.numpy().astype("uint8")))
            
            biou = np.mean(biou)
            
            train_loss_metric.update_state(anns, logits)
            train_miou_metric.append(miou)
            train_biou_metric.append(biou)
            if step == len_train_ds - 1:
                os.mkdir(f"output_images/train_ds/{epoch}")
                for i, pi in enumerate(pred_images):
                    f, x = plt.subplots(1, 3)
                    x[0].imshow(pi.numpy())
                    x[1].imshow(iou_anns[i].numpy())
                    x[2].imshow(imgs[i].numpy())
                    plt.savefig(f"output_images/train_ds/{epoch}/{i}_.png", dpi=200)
                break
        
        for step, (imgs, anns) in enumerate(val_ds):
            loss, logits = evaluate_step(m, imgs, anns, loss_fn)
            pred_images = tf.squeeze(tf.math.argmax(tf.nn.softmax(logits, axis=-1), axis=-1))
            iou_anns = tf.squeeze(tf.math.argmax(anns, axis=-1))
            miou = mean_iou(iou_anns, pred_images).numpy()
            biou = []
            for i, pi in enumerate(pred_images):
                biou.append(boundary_iou(iou_anns[i].numpy().astype("uint8"), pi.numpy().astype("uint8")))
            
            biou = np.mean(biou)
            
            val_loss_metric.update_state(anns, logits)
            val_miou_metric.append(miou)
            val_biou_metric.append(biou)
            if step == len_val_ds - 1:
                os.mkdir(f"output_images/val_ds/{epoch}")
                for i, pi in enumerate(pred_images):
                    f, x = plt.subplots(1, 3)
                    x[0].imshow(pi.numpy())
                    x[1].imshow(iou_anns[i].numpy())
                    x[2].imshow(imgs[i].numpy())
                    plt.savefig(f"output_images/val_ds/{epoch}/{i}_.png", dpi=200)
                break


        # Display metrics at the end of each epoch.
        train_loss = train_loss_metric.result()
        val_loss = val_loss_metric.result()
        train_miou = sum(train_miou_metric) / len(train_miou_metric)
        val_miou = sum(val_miou_metric) / len(val_miou_metric)
        train_biou = sum(train_biou_metric) / len(train_biou_metric)
        val_biou = sum(val_biou_metric) / len(val_biou_metric)
        print("Train loss: %.4f | Val loss: %.4f" % (float(train_loss), float(val_loss)))
        print("Train miou: %.4f | Val miou: %.4f" % (float(train_miou), float(val_miou)))
        print("Train biou: %.4f | Val biou: %.4f" % (float(train_biou), float(val_biou)))
        print()

        # Reset training metrics at the end of each epoch
        train_loss_metric.reset_states()
        val_loss_metric.reset_states()
        train_miou_metric = []
        val_miou_metric = []
        train_biou_metric = []
        val_biou_metric = []