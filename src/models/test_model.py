import datasets
from pathlib import Path
import yaml


import numpy as np
import torch
import torch.nn as nn
import random
import os

from transformers import Wav2Vec2FeatureExtractor, HubertModel
from torch.utils.data import Dataset, DataLoader
from torcheval.metrics.functional import multiclass_f1_score

import torch.nn.functional as F

from src import ROOT_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from src.models.Hubert_Classifier_model import HubertForAudioClassification

from pathlib import Path

import mlflow


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
    test_loader = DataLoader(dataset=audio_dataset["test"], batch_size=batch_size, shuffle=True, drop_last=True)

    print('Test set has {} instances'.format(len(audio_dataset["test"])))

    return test_loader

def test(loader, model, criterion, num_classes):
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

    print('Test: \tMean Loss: {:.6f} / F1-score {:.6f}'.format(global_epoch_loss/samples, F1_score))

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
    print("------- Testing of",params["algorithm_name"],"-------")

    # Set Mlflow experiment
    mlflow.set_tracking_uri('https://dagshub.com/armand-07/TAED2-Falsettos.mlflow')
    mlflow.log_param('mode', 'testing')
    mlflow.log_params(params)

    test_loader = data_loading (PROCESSED_DATA_DIR, params["batch_size"]) # data loading

    # ============== #
    # MODEL CREATION #
    # ============== #
    model = HubertForAudioClassification(adapter_hidden_size=params["adapter_hidden_size"])
    # Load the state dictionary and then assign it to the model
    PATH = os.path.join(MODELS_DIR,'final_model', '{}_bestmodel.pt'.format(params["algorithm_name"]))
    model.load_state_dict(torch.load(PATH), strict=False)
    
    # Define criterion
    criterion = nn.CrossEntropyLoss(reduction = 'mean')

    # Move the model to GPU
    if torch.cuda.is_available():
        print('Using CUDA with {0} GPUs'.format(torch.cuda.device_count()))
    model.cuda()

    # ============== #
    # MODEL TESTING  #
    # ============== #
    print("------------- Testing phase ----------------")

    test_loss, test_F1 = test(test_loader, model, criterion, params["num_classes"])
    mlflow.log_metric("Average Loss Test", test_loss)
    mlflow.log_metric("Best F1-score", test_F1)

    mlflow.pytorch.log_model(model, "model")


main()