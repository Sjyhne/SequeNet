# train.py

from models.metrics import mean_iou
from generator import create_dataset_generator, create_cityscapes_generator, create_dataset_from_model
from train_utils import calc_biou, remove_all_folders_in_path, store_images
from train_utils import display_and_store_metrics, save_best_model, calculate_sample_weight


from models.models.all_models import model_from_name

from train_utils import get_loss_func

import tensorflow as tf
from tqdm import tqdm
import numpy as np

import argparse
import json
import shutil
import os
import time

NUM_CLASSES = 2

def train_step(m, x, y, loss_func, cce_loss, optimizer, dist_map, args):
    with tf.GradientTape()  as tape:
        logits = m(x, training=True)
        softmaxed_logits = tf.nn.softmax(logits, axis=-1)
        loss_val = cce_loss(y, logits)
        if args.loss == "abl":
            abl_val = abl_func(y, logits, dist_map)
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

def evaluate_step(m, x, y, loss_func, dist_map, args):
    logits = m(x, training=False)
    softmaxed_logits = tf.nn.softmax(logits, axis=-1)
    if args.loss == "tfabl":
        loss_val = loss_func(y, logits, dist_map)
    else:
        loss_val = loss_func(y, logits)
    return loss_val, softmaxed_logits, logits

def train(args, train_ds, val_ds):
    
    args.learning_rate_decay = args.epochs * len(train_ds)
    
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        args.init_lr,
        args.epochs * len(train_ds),
        args.end_lr,
        power=0.2)
    
    if os.path.exists(os.path.join("model_output", "extra_" + args.model_type)):
        shutil.rmtree(os.path.join("model_output", "extra_" + args.model_type))
        
    os.mkdir(os.path.join("model_output", "extra_" + args.model_type))
    
    with open(os.path.join("model_output", "extra_" + args.model_type, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Add learning rate scheduler to the optimizer -- Believe that should work -- CosineWarmStart or something
    main_optimizer = tf.keras.optimizers.Adam(learning_rate=args.init_lr)
    main_loss_fn = get_loss_func(args.loss, args.label_smooth)
    main_model = model_from_name[args.model_type](args.num_classes, input_height=args.image_dim, input_width=args.image_dim)
        
    # Prepare the metrics.
    train_loss_metric = []
    val_loss_metric = []
    train_miou_metric = []
    val_miou_metric = []
    train_biou_metric = []
    val_biou_metric = []

    
    if os.path.exists(f"model_output/extra_ + {args.model_type}/output_images/train/"):
        remove_all_folders_in_path(f"model_output/extra_ + {args.model_type}/output_images/train/")
    if os.path.exists(f"model_output/extra_ + {args.model_type}/output_images/val/"):
        remove_all_folders_in_path(f"model_output/extra_ + {args.model_type}/output_images/val/")

    epochs = args.epochs
    
    best_loss_value = 100_000
    extra_best_loss_value = 100_000
    
    print(f"Starting to train for {epochs} epochs")
    start = time.time()
    for epoch in range(epochs):
        print(f"Epoch: {epoch}/{epochs}")
        print()
        
        for step, (imgs, anns, names, dist_map) in tqdm(enumerate(train_ds), total=len(train_ds)):
            loss, softmaxed_logits, logits = train_step(main_model, imgs, anns, main_loss_fn, main_optimizer, dist_map, args)
            miou, biou = compute_metrics(softmaxed_logits, anns)
            
            train_loss_metric.append(loss)
            train_miou_metric.append(miou)
            train_biou_metric.append(biou)
            if step == len(train_ds) - 1:
                store_images(f"model_output/extra_{args.model_type}/output_images/train/{epoch}", anns, imgs, softmaxed_logits, names)
        
        for step, (imgs, anns, names, dist_map) in enumerate(val_ds):
            loss, softmaxed_logits, logits = evaluate_step(main_model, imgs, anns, main_loss_fn, dist_map, args)
            miou, biou = compute_metrics(softmaxed_logits, anns)

            val_loss_metric.append(loss)
            val_miou_metric.append(miou)
            val_biou_metric.append(biou)
            if step == len(val_ds) - 1:
                store_images(f"model_output/extra_{args.model_type}/output_images/val/{epoch}", anns, imgs, softmaxed_logits, names)
        
        print("Current time taken since start:", round(time.time() - start, 3), "seconds")
        print("Estimated total time:", ((time.time() - start)/(epoch + 1)) * epochs, "seconds")
        print("Current lr:", round(learning_rate_fn(epoch).numpy(), 7))
        
        display_and_store_metrics(
            train_loss_metric, val_loss_metric,
            train_miou_metric, val_miou_metric,
            train_biou_metric, val_biou_metric,
            "extra", args
        )
        
        best_loss_value = save_best_model(main_model, round((sum(val_loss_metric)/len(val_loss_metric)).numpy(), 3), best_loss_value, epoch, "extra", args)

        # Reset training metrics at the end of each epoch
        train_loss_metric = []
        val_loss_metric = []
        train_miou_metric = []
        val_miou_metric = []
        train_biou_metric = []
        val_biou_metric = []


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
    parser.add_argument("--loss", type=str, default="cce", help="The loss function to be used for the main segmentation network")
    parser.add_argument("--label_smooth", type=float, default=0.0, help="The label smoothing value for the loss function for the main network")
    parser.add_argument("--load_model", type=str, default="", help="The path to the model that should generate the dataset, if dataset not exist")
    parser.add_argument("--overwrite_dataset", type=bool, default=False, help="Whether we should overwrite the data if it already exists")
    parser.add_argument("--end_lr", type=float, default=1e-5, help="Finishing value of the learning rate when the training is finished")
    parser.add_argument("--create_dist", type=bool, default=False, help="Whether or not to create distmaps for the dataset")


    args = parser.parse_args()
    
    # load main model here and perform dataset generation
    
    if args.dataset == "lba":
        train_ds = create_dataset_generator(args.data_path, "train", batch_size=args.batch_size, data_percentage=args.data_percentage, create_dist=True)
        val_ds = create_dataset_generator(args.data_path, "val", batch_size=args.batch_size, data_percentage=args.data_percentage, create_dist=True)
        test_ds = create_dataset_generator(args.data_path, "test", batch_size=args.batch_size, data_percentage=args.data_percentage, create_dist=True)
    elif args.dataset == "cityscapes":
        train_ds = create_cityscapes_generator("train")
        val_ds = create_cityscapes_generator("val")
        test_ds = create_cityscapes_generator("test")
    
    if args.load_model != "":
        train_ds = create_dataset_from_model(train_ds, "train", args)
        val_ds = create_dataset_from_model(val_ds, "val", args)
        test_ds = create_dataset_from_model(test_ds, "test", args)
    else:
        print("Not supported anything other than using load_model")
        exit()
    
    print(len(train_ds))
    
    train(args, train_ds, val_ds)