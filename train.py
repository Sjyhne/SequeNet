# train.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

from models.metrics import mean_iou
from generator import create_dataset_generator, create_cityscapes_generator
from train_utils import calc_biou, remove_all_folders_in_path, store_images
from train_utils import display_and_store_metrics, save_best_model, calculate_sample_weight


from models.models.all_models import model_from_name

from train_utils import get_loss_func

import cv2

import tensorflow as tf
tf.debugging.experimental.disable_dump_debug_info()
from tqdm import tqdm
import numpy as np

import argparse
import json
import shutil
import time

NUM_CLASSES = 2

def train_step(m, x, y, loss_func, cce_loss, optimizer, dist_map, args):
    with tf.GradientTape()  as tape:
        logits = m(x, training=True)
        softmaxed_logits = tf.nn.softmax(logits, axis=-1)
        loss_val = cce_loss(y, logits)
        if args.main_loss == "abl":
            abl_val = loss_func(y, logits, dist_map)
            loss_val = loss_val + abl_val

    # Use the gradient tape to automatically retrieve
    # the gradients of the trainable variables with respect to the loss.
    grads = tape.gradient(loss_val, m.trainable_variables)
    
    # Run one step of gradient descent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, m.trainable_variables))
    return loss_val, softmaxed_logits, logits

def compute_metrics(softmaxed_logits, anns):
    pred_images = tf.math.argmax(softmaxed_logits, axis=-1)
    iou_anns = tf.math.argmax(anns, axis=-1)
    miou = mean_iou(iou_anns, pred_images).numpy()
    biou = calc_biou(pred_images, iou_anns)
    
    return miou, biou

def evaluate_step(m, x, y, loss_func, cce_loss, dist_map, args):
    logits = m(x, training=False)
    softmaxed_logits = tf.nn.softmax(logits, axis=-1)
    loss_val = cce_loss(y, logits)
    if args.main_loss == "abl":
        abl_val = loss_func(y, logits, dist_map)
        loss_val = loss_val + abl_val
        
    return loss_val, softmaxed_logits, logits

def train(args, train_ds, val_ds):
    
    args.learning_rate_decay = args.epochs * len(train_ds)
    
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        args.init_lr,
        args.epochs * len(train_ds),
        args.end_lr,
        power=0.5)
    
    if os.path.exists(os.path.join("model_output", "main_" + args.model_type)):
        shutil.rmtree(os.path.join("model_output", "main_" + args.model_type))
        
    os.mkdir(os.path.join("model_output", "main_" + args.model_type))
    
    with open(os.path.join("model_output", "main_" + args.model_type, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Add learning rate scheduler to the optimizer -- Believe that should work -- CosineWarmStart or something
    main_optimizer = tf.keras.optimizers.Adam(learning_rate=args.init_lr)
    main_loss_fn = get_loss_func(args.main_loss, args.label_smooth)
    cce_loss = get_loss_func("cce", 0.0)
    if args.model_type == "msrf":
        main_model = model_from_name[args.model_type]((512, 512, 3), (512, 512, 3))
    else:
        main_model = model_from_name[args.model_type](args.num_classes, input_height=args.image_dim, input_width=args.image_dim)
    
    if args.extra_model == True:
        extra_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
        extra_loss_fn = get_loss_func(args.extra_loss, args.extra_label_smooth)
        extra_model = model_from_name[args.extra_model_type](args.num_classes, input_height=args.image_dim, input_width=args.image_dim, channels=args.num_channels)
        
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
    
    if os.path.exists(f"model_output/main_{args.model_type}/output_images/train/"):
        remove_all_folders_in_path(f"model_output/main_{args.model_type}/output_images/train/")
    if os.path.exists(f"model_output/main_{args.model_type}/output_images/val/"):
        remove_all_folders_in_path(f"model_output/main_{args.model_type}/output_images/val/")

    epochs = args.epochs
    
    best_loss_value = 100_000
    extra_best_loss_value = 100_000
    
    print(f"Starting to train for {epochs} epochs")
    start = time.time()
    for epoch in range(epochs):
        print(f"Epoch: {epoch}/{epochs}")
        print()
        for step, (imgs, anns, names, dist_map) in tqdm(enumerate(train_ds), total=len(train_ds)):
            loss, softmaxed_logits, logits = train_step(main_model, imgs, anns, main_loss_fn, cce_loss, main_optimizer, dist_map, args)
            miou, biou = compute_metrics(softmaxed_logits, anns)
            if args.extra_model == True and epoch >= main_model_pretraining:
                logits = tf.stop_gradient(softmaxed_logits)
                logits = (tf.cast(imgs, dtype=tf.float32) * tf.expand_dims(tf.clip_by_value(logits[:, :, :, 1], 0.3, 1.0), axis=-1))/255
                extra_loss, extra_softmaxed_logits, extra_logits = train_step(extra_model, logits, anns, extra_loss_fn, extra_optimizer)
                miou, biou = compute_metrics(extra_softmaxed_logits, anns)
                extra_train_loss_metric.append(extra_loss)
                extra_train_miou_metric.append(miou)
                extra_train_biou_metric.append(biou)
                if step == len(train_ds) - 1:
                    store_images(f"model_output/extra_{args.model_type}/output_images/train/{epoch}", anns, imgs, softmaxed_logits, extra_softmaxed_logits)

            train_loss_metric.append(loss)
            train_miou_metric.append(miou)
            train_biou_metric.append(biou)
            #main_optimizer.learning_rate = learning_rate_fn((epoch * len(train_ds)) + step)
            #print(main_optimizer.learning_rate)
            if args.extra_model == False or epoch < main_model_pretraining:
                if step == len(train_ds) - 1:
                    store_images(f"model_output/main_{args.model_type}/output_images/train/{epoch}", anns, imgs, softmaxed_logits, names)

        
        for step, (imgs, anns, names, dist_map) in enumerate(val_ds):
            loss, softmaxed_logits, logits = evaluate_step(main_model, imgs, anns, main_loss_fn, cce_loss, dist_map, args)
            miou, biou = compute_metrics(softmaxed_logits, anns)
            if args.extra_model == True and epoch >= main_model_pretraining:
                logits = tf.stop_gradient(softmaxed_logits)
                logits = (tf.cast(imgs, dtype=tf.float32) * tf.expand_dims(tf.clip_by_value(logits[:, :, :, 1], 0.2, 1.0), axis=-1))/255
                extra_loss, extra_softmaxed_logits, extra_logits = train_step(extra_model, logits, anns, extra_loss_fn, extra_optimizer)
                miou, biou = compute_metrics(extra_softmaxed_logits, anns)
                extra_val_loss_metric.append(extra_loss)
                extra_val_miou_metric.append(miou)
                extra_val_biou_metric.append(biou)
                if step == len(val_ds) - 1:
                    store_images(f"model_output/extra_{args.model_type}/output_images/val/{epoch}", anns, imgs, softmaxed_logits, extra_softmaxed_logits)

            val_loss_metric.append(loss)
            val_miou_metric.append(miou)
            val_biou_metric.append(biou)
            if args.extra_model == False or epoch < main_model_pretraining:
                if step == len(val_ds) - 1:
                    store_images(f"model_output/main_{args.model_type}/output_images/val/{epoch}", anns, imgs, softmaxed_logits, names)
        
        time_taken = round(time.time() - start, 3)
        print("Current time taken since start:", time_taken, "seconds or", round(time_taken/60, 3), "minutes or", round(time_taken/(60*60), 3), "hours")
        print("Estimated total time:", round(time_taken/(epoch + 1), 3) * epochs, "seconds or", round(time_taken/((epoch + 1) * 60), 3) * epochs, "minutes or", round(time_taken/((epoch + 1) * 60 * 60), 3) * epochs, "hours")
        
        display_and_store_metrics(
            train_loss_metric, val_loss_metric,
            train_miou_metric, val_miou_metric,
            train_biou_metric, val_biou_metric,
            "main", args
        )
        
        if args.extra_model == True and epoch >= main_model_pretraining:
            display_and_store_metrics(
                extra_train_loss_metric, extra_val_loss_metric,
                extra_train_miou_metric, extra_val_miou_metric,
                extra_train_biou_metric, extra_val_biou_metric,
                "extra", args
            )
        
        best_loss_value = save_best_model(main_model, round((sum(val_loss_metric)/len(val_loss_metric)).cpu().numpy(), 5), best_loss_value, epoch, "main", args)
        if args.extra_model == True and epoch >= main_model_pretraining:
            extra_best_loss_value = save_best_model(extra_model, round((sum(extra_val_loss_metric)/len(extra_val_loss_metric)).cpu().numpy(), 5), extra_best_loss_value, epoch, "extra", args)

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
    parser.add_argument("--main_loss", type=str, default="cce", help="The loss function to be used for the main segmentation network")
    parser.add_argument("--extra_loss", type=str, default="cce", help="The loss function to be used for the extra segmentation network")
    parser.add_argument("--main_label_smooth", type=float, default=0.0, help="The label smoothing value for the loss function for the main network")
    parser.add_argument("--extra_label_smooth", type=float, default=0.0, help="The label smoothing value for the loss function for the extra network")
    parser.add_argument("--end_lr", type=float, default=1e-5, help="Finishing value of the learning rate when the training is finished")
    parser.add_argument("--label_smooth", type=float, default=0, help="Label-smoothing for the losses, only works with cce and abl(?)")

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