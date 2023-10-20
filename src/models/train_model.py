import datasets
from pathlib import Path
import yaml

import os
import time

import numpy as np
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
import random

from transformers import Wav2Vec2FeatureExtractor, HubertModel
from torch.utils.data import Dataset, DataLoader
from torcheval.metrics.functional import multiclass_f1_score

import torch.nn.functional as F

from src import ROOT_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from src.models.Hubert_Classifier_model import HubertForAudioClassification

from pathlib import Path

import mlflow
from codecarbon import EmissionsTracker


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def data_loading(input_folder_path, batch_size):
    # Load the dataset
    print("Loading dataset from disk...")
    audio_dataset = datasets.load_from_disk(input_folder_path)
    audio_dataset.set_format(type='torch', columns=['key', 'features', 'label'])
     # Create the dataloaders
    train_loader = DataLoader(dataset=audio_dataset["train"], batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=audio_dataset["validation"], batch_size=batch_size, shuffle=True, drop_last=True)

    print('Training set has {} instances'.format(len(audio_dataset["train"])))
    print('Validation set has {} instances'.format(len(audio_dataset["validation"])))

    return train_loader, val_loader


def train(loader, model, criterion, optimizer, epoch, log_interval, verbose=True):
    model.train()
    global_epoch_loss = 0
    samples = 0
    for batch_idx, batch in enumerate(loader):
        data, target = batch["features"].cuda(), batch["label"].cuda()
        optimizer.zero_grad()
        logits = model(data)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()
        global_epoch_loss += loss.data.item() * len(target)
        samples += len(target)

        if verbose and (batch_idx % log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, samples, len(loader.dataset), 100*samples/len(loader.dataset), global_epoch_loss/samples))

    return global_epoch_loss / samples


def val(loader, model, criterion, epoch, num_classes):
    model.eval()
    global_epoch_loss = 0
    total_preds = torch.Tensor()
    total_targets = torch.Tensor()
    samples = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # Get the the outputs
            data, target = batch["features"].cuda(), batch["label"].cuda()
            logits = model(data)
            output = F.log_softmax(logits, dim=1)
            # Get the indices of the maximum values along the class dimension
            predicted_classes = torch.argmax(output, dim=1)

            # Append the outputs and the targets to the tensors and compute metrics
            total_preds = torch.cat((total_preds, predicted_classes.cpu()), dim=0)
            total_targets = torch.cat((total_targets, torch.argmax(target, dim=1).cpu()), dim=0)

            loss = criterion(logits, target)
            global_epoch_loss += loss.data.item() * len(target)
            samples += len(target)


    F1_score = multiclass_f1_score(input = total_preds, target = total_targets, num_classes = num_classes)

    print('Validation Epoch: {} \tMean Loss: {:.6f} / F1-score {:.6f}'.format(epoch, global_epoch_loss/samples, F1_score))

    return global_epoch_loss / samples, F1_score



def main():
    # Path of the parameters file
    params_path = Path("params.yaml")
    params_path = ROOT_DIR / params_path

    # Read data preparation parameters
    with open(params_path, "r") as params_file:
        try:
            params = yaml.safe_load(params_file)
            params = params["model"]
        except yaml.YAMLError as exc:
            print(exc)


    seed_everything(params["random_state"])
    print("------- Training of",params["algorithm_name"],"-------")

    # Set Mlflow experiment
    mlflow.set_tracking_uri('https://dagshub.com/armand-07/TAED2-Falsettos.mlflow')
    mlflow.log_param('mode', 'training')
    mlflow.log_params(params)

    train_loader, val_loader = data_loading (PROCESSED_DATA_DIR, params["batch_size"]) # data loading

    # ============== #
    # MODEL CREATION #
    # ============== #
    model = HubertForAudioClassification(adapter_hidden_size = params["adapter_hidden_size"])
    # partial freeze of wac2vec parameters. Only feature_projection parameters are fine tuned
    model.freeze_feature_encoder()
    for param in model.hubert.encoder.parameters():
        param.requires_grad = False

    # Define criterion
    criterion = nn.CrossEntropyLoss(reduction = 'mean')
    # Define optimizer
    if params["optimizer"].lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=params["lr"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=params["lr"], momentum=params["momentum"])

    

    # Move the model to GPU
    if torch.cuda.is_available():
        print('Using CUDA with {0} GPUs'.format(torch.cuda.device_count()))
    model.cuda()



    # ============== #
    # MODEL TRAINING #
    # ============== #
    print("------------- Training phase ----------------")
    # Define the emissions tracker
    tracker = EmissionsTracker(measure_power_secs=10, output_dir=os.path.join(MODELS_DIR, 'final_model'), log_level= "warning") # measure power every 10 seconds
    tracker.start()

    #Define saving checkpoints path
    checkpoint_path = Path(str(MODELS_DIR)+"/checkpoints")
    if not os.path.exists(checkpoint_path):
       os.mkdir(checkpoint_path)

    # Define the training parameters
    best_val_F1 = 0.0
    iteration = 0
    epoch = 1
    best_epoch = epoch

    # training with early stopping
    t0 = time.time()
    while (epoch < params["epochs"] + 1) and (iteration < params["patience"]):
        train_loss = train(train_loader, model, criterion, optimizer, epoch, params["log_interval"])
        val_loss, val_F1 = val(val_loader, model, criterion, epoch, params["num_classes"])

        mlflow.log_metric("Loss evolution in training", train_loss, step=epoch)
        mlflow.log_metric("Loss evolution in validation", val_loss, step=epoch)
        mlflow.log_metric("F1-score in validation", val_F1, step=epoch)

        torch.save(model.state_dict(), str(checkpoint_path)+'/model_{:03d}.pt'.format(epoch))

        if val_F1 <= best_val_F1:
            iteration += 1
            print('F1-score was not improved, iteration {0}'.format(str(iteration)))
        else:
            print('Saving state')
            iteration = 0
            best_val_F1 = val_F1
            best_epoch = epoch
            state = {
                'valid_auc': val_F1,
                'valid_loss': val_loss,
                'epoch': epoch,
            }
            torch.save(state, str(checkpoint_path)+'/ckpt.pt')
        epoch += 1
        print(f'Elapsed seconds: ({time.time() - t0:.0f}s)')
    print(f'Best F1-Score: {best_val_F1*100:.1f}% on epoch {best_epoch}')


    # ============== #
    # MODEL SAVING   #
    # ============== #
    print("-------------- Model saving -----------------")
    # Get best epoch and model
    state = torch.load(str(checkpoint_path)+'/ckpt.pt')
    epoch = state['epoch']
    model.load_state_dict(torch.load(str(checkpoint_path)+'/model_{:03d}.pt'.format(epoch)))

    # Save model
    final_model_save_path = os.path.join(MODELS_DIR, 'final_model')
    if not os.path.exists(final_model_save_path):
       os.mkdir(final_model_save_path)
    print(os.path.join(MODELS_DIR, 'final_model', '{}_bestmodel_{:03d}.pt'.format(params["algorithm_name"], epoch)))
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'final_model', '{}_bestmodel.pt'.format(params["algorithm_name"])))

    emissions: float = tracker.stop()
    # Log metrics to Mlflow
    mlflow.log_metric("Emissions in CO2 kg", float(emissions))
    mlflow.log_metric("Best F1-score", best_val_F1)

    # Log artifacts to Mlflow
    mlflow.log_artifact(os.path.join(MODELS_DIR, 'final_model', 'emissions.csv'), 'metrics')
    mlflow.pytorch.log_model(model, "model")

main()
