import torch
from torchmetrics import JaccardIndex, Accuracy
from tqdm import tqdm
import numpy as np

from train_utils import aggregate_metrics, display_and_store_metrics, get_loss, get_optim, calc_biou, save_best_model, store_images, iou_pytorch
from models.model_hub import get_model
from generator import create_dataset_generator

import argparse
import json
import shutil
import os
import time

JACCARD_INDEX = JaccardIndex(num_classes=2, ignore_index=255).to("cuda:0")
ACCURACY = Accuracy(num_classes=2).to("cuda:0")

def compute_metrics(logits, labs):
    pred_masks = torch.argmax(torch.nn.functional.softmax(logits, dim=1), dim=1)
    lab_masks = labs
    miou = iou_pytorch(pred_masks, lab_masks)
    biou = calc_biou(pred_masks, lab_masks)
    #acc = ACCURACY(pred_masks, lab_masks)
    
    #acc = acc.cpu().numpy()
    
    acc = np.nan
    
    miou = miou.cpu().numpy()
    
    if acc == np.nan:
        acc = 1e-6
    if miou == np.nan:
        miou = 1e-6
    if biou == np.nan:
        biou = 1e-6

    return acc, miou, biou


def train_step(model, batch, loss_fn, optim, args, save=False):
    imgs = batch["img"].to(args.device)
    labs = batch["lab"].to(args.device)
    
    imgs = torch.permute(imgs, (0, 3, 1, 2))
    
    optim.zero_grad()

    logits = model(imgs)
    if args.loss == "abl":
        loss = loss_fn(logits, labs, batch["dist_map"].to(args.device), save)
    else:
        loss = loss_fn(logits, labs)
    loss.backward()
    optim.step()
    
    logits = logits.detach()
    
    acc, miou, biou = compute_metrics(logits, labs)

    return {"loss": loss.item(), "miou": miou, "biou": biou, "acc": acc, "logits": logits}

def eval_step(model, batch, loss_fn, args):
    imgs = batch["img"].to(args.device)
    labs = batch["lab"].to(args.device)

    imgs = torch.permute(imgs, (0, 3, 1, 2))

    logits = model(imgs)
    if args.loss == "abl":
        loss = loss_fn(logits, labs, batch["dist_map"].to(args.device))
    else:
        loss = loss_fn(logits, labs)
    
    logits = logits.detach()
    acc, miou, biou = compute_metrics(logits, labs)

    return {"loss": loss.item(), "miou": miou, "biou": biou, "acc": acc, "logits": logits}
    

def train(args, train_ds, val_ds):

    output_path = os.path.join("model_output", args.training_mode + "_" + args.model + "_e" + str(args.epochs) + "_" + str(args.image_dim) + "_" + args.loss)
    print("Output path:", output_path)
    
    args.output_path = output_path

    if os.path.exists(output_path):
        ans = None
        while ans not in ["y", "n"]:
            ans = input(f"Sure you want to delete contents in {output_path}? ")
        if ans == "y":
            print("Removing the contents")
            shutil.rmtree(output_path)
            print("Successfully removed the contents")
        else:
            raise RuntimeError("Exiting based on the command specified by the user")
    
    os.makedirs(output_path)

    print("Args:", args)

    with open(os.path.join(output_path, "args.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    # TODO: Find a good way to combine the loss functions -- maybe just return a function
    # That calculates the loss function?
    model = get_model(args).to(args.device)
    optim = get_optim(args.optim)(model.parameters(), lr=args.lr)
    loss_fn = get_loss(args)
    
    if args.load_from != None:
        try:
            model.load_state_dict(torch.load(args.load_from))
            print("Successfully loaded model from state dict:", args.load_from)
        except Exception as e:
            print(e)

    best_loss_value = None

    print("-----------------------------------------")
    print(f"---- Starting training for {args.epochs} epochs ----")
    print("-----------------------------------------")

    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")

        train_metrics = {
            "loss": [],
            "acc": [],
            "miou": [],
            "biou": []
        }

        eval_metrics = {
            "loss": [],
            "acc": [],
            "miou": [],
            "biou": []
        }


        model.train()
        for step, batch in tqdm(enumerate(train_ds), total=len(train_ds), leave=False):
            if step == len(train_ds) - 1:
                res = train_step(model, batch, loss_fn, optim, args, save=True)
            else:
                res = train_step(model, batch, loss_fn, optim, args, save=False)
            train_metrics = aggregate_metrics(train_metrics, res)
            if step == len(train_ds) - 1:
                store_images(os.path.join(output_path, "train", str(epoch + 1)), batch, res["logits"])
            
        for key in train_metrics.keys():
            train_metrics[key] = np.mean(train_metrics[key])

        model.eval()
        for step, batch in tqdm(enumerate(val_ds), total=len(val_ds), leave=False):
            res = eval_step(model, batch, loss_fn, args)
            eval_metrics = aggregate_metrics(eval_metrics, res)
            if step == len(val_ds) - 1:
                store_images(os.path.join(output_path, "val", str(epoch + 1)), batch, res["logits"])
            
        for key in eval_metrics.keys():
            eval_metrics[key] = np.mean(eval_metrics[key])

        time_taken = round(time.time() - start_time, 3)
        print("Current time taken since start:", time_taken, "seconds or", round(time_taken/60, 3), "minutes or", round(time_taken/(60*60), 3), "hours")
        print("Estimated total time:", round(time_taken/(epoch + 1), 3) * args.epochs, "seconds or", round(time_taken/((epoch + 1) * 60), 3) * args.epochs, "minutes or", round(time_taken/((epoch + 1) * 60 * 60), 3) * args.epochs, "hours")

        display_and_store_metrics(train_metrics, eval_metrics, args)

        best_loss_value = save_best_model(model, optim, round(eval_metrics["loss"], 6), best_loss_value, epoch + 1, args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add custom arguments for the training of the model(s)")
    # TODO: This should be revised -- maybe added in a config file or something
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs for training")
    parser.add_argument("--lr", type=float, default=5e-4, help="The initial learning rate")
    parser.add_argument("--image_dim", type=int, default=512, help="The dimensions of the input image")
    parser.add_argument("--num_channels", type=int, default=3, help="Number of channels in input image")
    parser.add_argument("--num_classes", type=int, default=2, help="The number of classes to predict")
    parser.add_argument("--model", type=str, default="unet", help="The model type to be trained")
    parser.add_argument("--batch_size", type=int, default=8, help="The batchsize used for the training")
    parser.add_argument("--data_percentage", type=float, default=1.0, help="The percentage size of data to be used during training")
    parser.add_argument("--dataset", type=str, default="lba", help="The dataset of choosing for the training and/or evaluation")
    parser.add_argument("--loss", type=str, default="cce", help="The loss function to be used for the main segmentation network")
    parser.add_argument("--label_smooth", type=float, default=0, help="Label-smoothing for the losses, only works with cce and abl(?)")
    parser.add_argument("--training_mode", type=str, default="primary", help="Is it the primary or secondary network being trained?")
    parser.add_argument("--optim", type=str, default="adam", help="Which optimizer to be used for the training")
    parser.add_argument("--device", type=str, default="cuda:0", help="What type of device to be used for training")
    parser.add_argument("--abl_weight", type=float, default=1.0, help="What the loss value of the abl should should be weighted with")
    parser.add_argument("--load_from", type=str, default=None, help="Path to .pt file so that we can load a pretrained version of the model")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--four_channels", type=bool, default=False)
    
    args = parser.parse_args()

    
    print("Args:", args)

    if args.dataset == "lba":
        train_ds = torch.utils.data.DataLoader(create_dataset_generator(args.data_path, "train", batch_size=args.batch_size, data_percentage=args.data_percentage, four_channels=args.four_channels), shuffle=True, batch_size=args.batch_size)
        val_ds = torch.utils.data.DataLoader(create_dataset_generator(args.data_path, "val", batch_size=args.batch_size, data_percentage=args.data_percentage, four_channels=args.four_channels), shuffle=False, batch_size=args.batch_size)
        test_ds = torch.utils.data.DataLoader(create_dataset_generator(args.data_path, "test", batch_size=args.batch_size, data_percentage=args.data_percentage, four_channels=args.four_channels), shuffle=False, batch_size=args.batch_size)

    print("Dataset sizes")
    print("Train:\t", len(train_ds))
    print("Val:\t", len(val_ds))
    print("Test:\t", len(test_ds))
    
    train(args, train_ds, val_ds)