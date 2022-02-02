# train.py

from models.metrics import mean_iou
from generator import create_dataset_generator, create_cityscapes_generator
from train_utils import calc_biou, remove_all_folders_in_path, store_images
from train_utils import display_and_store_metrics, save_best_model, calculate_sample_weight


from models.models.all_models import model_from_name

import tensorflow as tf
from tqdm import tqdm
import numpy as np

import argparse
import os

NUM_CLASSES = 2

def train_step(m, x, y, loss_func, optimizer):
    with tf.GradientTape()  as tape:
        logits = m(x, training=True)
        softmaxed_logits = tf.nn.softmax(logits, axis=-1)
        loss_val = loss_func(y, softmaxed_logits)
        print("loss_val:", loss_val)

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss_val, m.trainable_variables)
    
    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, m.trainable_variables))
    return loss_val, softmaxed_logits, logits

def compute_metrics(softmaxed_logits, anns, imgs, step, epoch, model_type):
    pred_images = tf.math.argmax(softmaxed_logits, axis=-1)
    iou_anns = tf.squeeze(anns, axis=-1)
    miou = mean_iou(iou_anns, pred_images).numpy()
    biou = calc_biou(pred_images, iou_anns)
    
    if step == len(train_ds) - 1:
        store_images(pred_images, iou_anns, imgs, f"output_images/train_ds/{model_type}/{epoch}")

    return miou, biou

def evaluate_step(m, x, y, loss_func):
    logits = m(x, training=False)
    softmaxed_logits = tf.nn.softmax(logits, axis=-1)
    loss_val = loss_func(y, logits)
    return loss_val, softmaxed_logits, logits

def train(args, train_ds, val_ds):
    
    # Add learning rate scheduler to the optimizer -- Believe that should work -- CosineWarmStart or something
    main_optimizer = tf.keras.optimizers.SGD(learning_rate=args.init_lr)
    main_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    main_model = model_from_name[args.model_type](args.num_classes, input_height=args.image_dim, input_width=args.image_dim)
    
    if args.extra_model:
        extra_optimizer = tf.keras.optimizers.SGD(learning_rate=args.init_lr)
        extra_loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        extra_model = model_from_name[args.extra_model_type](args.num_classes, input_height=args.image_dim, input_width=args.image_dim, channels=args.num_classes)
        
    # Prepare the metrics.
    train_loss_metric = []
    val_loss_metric = []
    train_miou_metric = []
    val_miou_metric = []
    train_biou_metric = []
    val_biou_metric = []
    
    extra_train_loss_metric = []
    extra_val_loss_metric = []
    extra_train_miou_metric = []
    extra_val_miou_metric = []
    extra_train_biou_metric = []
    extra_val_biou_metric = []

    remove_all_folders_in_path("output_images/train_ds/")
    remove_all_folders_in_path("output_images/val_ds/")

    epochs = args.epochs
    
    best_loss_value = 10_000
    extra_best_loss_value = 10_000
    
    print(f"Starting to train for {epochs} epochs")
    for epoch in range(epochs):
        print(f"Epoch: {epoch}/{epochs}")
        print()
        for step, (imgs, anns) in tqdm(enumerate(train_ds), total=len(train_ds)):
            loss, softmaxed_logits, logits = train_step(main_model, imgs, anns, main_loss_fn, main_optimizer)
            if args.extra_model:
                extra_loss, extra_softmaxed_logits, extra_logits = train_step(extra_model, logits, anns, extra_loss_fn, extra_optimizer)
                miou, biou = compute_metrics(extra_softmaxed_logits, anns, imgs, step, epoch, "extra")
                extra_train_loss_metric.append(extra_loss)
                extra_train_miou_metric.append(miou)
                extra_train_biou_metric.append(biou)

            miou, biou = compute_metrics(softmaxed_logits, anns, imgs, step, epoch, "main")
            train_loss_metric.append(loss)
            train_miou_metric.append(miou)
            train_biou_metric.append(biou)

        
        for step, (imgs, anns) in enumerate(val_ds):
            loss, softmaxed_logits, logits = evaluate_step(main_model, imgs, anns, main_loss_fn)
            if args.extra_model:
                extra_loss, extra_softmaxed_logits, extra_logits = train_step(extra_model, logits, anns, extra_loss_fn, extra_optimizer)
                miou, biou = compute_metrics(extra_softmaxed_logits, anns, imgs, step, epoch)
                extra_val_loss_metric.append(extra_loss)
                extra_val_miou_metric.append(miou)
                extra_val_biou_metric.append(biou)

            miou, biou = compute_metrics(softmaxed_logits, anns, imgs, step, epoch)
            val_loss_metric.append(loss)
            val_miou_metric.append(miou)
            val_biou_metric.append(biou)

        display_and_store_metrics(
            train_loss_metric, val_loss_metric,
            train_miou_metric, val_miou_metric,
            train_biou_metric, val_biou_metric
        )
        
        best_loss_value = save_best_model(main_model, np.mean(val_loss_metric), best_loss_value, epoch)
        extra_best_loss_value = save_best_model(extra_model, np.mean(extra_val_loss_metric), extra_best_loss_value, epoch)

        # Reset training metrics at the end of each epoch
        train_loss_metric = []
        val_loss_metric = []
        train_miou_metric = []
        val_miou_metric = []
        train_biou_metric = []
        val_biou_metric = []

        extra_train_loss_metric = []
        extra_val_loss_metric = []
        extra_train_miou_metric = []
        extra_val_miou_metric = []
        extra_train_biou_metric = []
        extra_val_biou_metric = []


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Add custom arguments for the training of the model(s)")

    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--init_lr", type=float, default=1e-3, help="The initial learning rate")
    parser.add_argument("--image_dim", type=int, default=512, help="The dimensions of the input image")
    parser.add_argument("--num_channels", type=int, default=3, help="Number of channels in input image")
    parser.add_argument("--num_classes", type=int, default=2, help="The number of classes to predict")
    parser.add_argument("--model_type", type=str, default="unet", help="The model type to be trained", choices=list(model_from_name.keys()))
    parser.add_argument("--batch_size", type=int, default=8, help="The batchsize used for the training")
    parser.add_argument("--data_path", type=str, default="data/large_building_area/img_dir", help="Path to data used for training")
    parser.add_argument("--data_percentage", type=float, default=1.0, help="The percentage size of data to be used during training")
    parser.add_argument("--dataset", type=str, default="lba", help="The dataset of choosing for the training and/or evaluation")
    parser.add_argument("--extra_model", type=bool, default=False, help="Whether to use the extra neural network model")
    parser.add_argument("--extra_model_type", type=str, default="fcn_8", help="What type of model the extra neural network should be")

    args = parser.parse_args()

    if args.dataset == "lba":
        train_ds = create_dataset_generator(args.data_path, "train", batch_size=args.batch_size, data_percentage=args.data_percentage)
        val_ds = create_dataset_generator(args.data_path, "val", batch_size=args.batch_size, data_percentage=args.data_percentage)
        test_ds = create_dataset_generator(args.data_path, "test", batch_size=args.batch_size, data_percentage=args.data_percentage)
    elif args.dataset == "cityscapes":
        train_ds = create_cityscapes_generator("train")
        val_ds = create_cityscapes_generator("val")
        test_ds = create_cityscapes_generator("test")

    train(args, train_ds, val_ds)