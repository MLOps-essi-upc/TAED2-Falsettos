"""
Given a trained model, test it and give its F1 score.
"""

from pathlib import Path
import os

import random
import numpy as np
import datasets
import yaml
import mlflow

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from torcheval.metrics.functional import multiclass_f1_score

from src import ROOT_DIR, MODELS_DIR, PROCESSED_DATA_DIR
from src.models.Hubert_Classifier_model import HubertForAudioClassification


def seed_everything(seed):
    """Set seeds to allow reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def data_loading(input_folder_path, batch_size):
    """Load the necessary data for evaluating the model."""
    # Load the dataset
    print("Loading dataset from disk...")
    audio_dataset = datasets.load_from_disk(input_folder_path)
    audio_dataset.set_format(type='torch', columns=['key', 'features', 'label'])
    # Create the dataloaders
    test_loader = DataLoader(
        dataset=audio_dataset["test"], batch_size=batch_size, shuffle=True, drop_last=True
    )
    print(f'Test set has {len(audio_dataset["test"])} instances')
    return test_loader


def eval_model(loader, model, criterion, num_classes):
    """Define the model evaluation."""
    model.eval()
    global_epoch_loss = 0
    total_preds = torch.Tensor()
    total_targets = torch.Tensor()
    samples = 0
    with torch.no_grad():
        for _, batch in enumerate(loader):
            # Get the outputs
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

    f1_score = multiclass_f1_score(input=total_preds, target=total_targets, num_classes=num_classes)

    print(f'Test: \tMean Loss: {global_epoch_loss / samples:.6f} / F1-score {f1_score:.6f}')

    return global_epoch_loss / samples, f1_score


def main():
    """Call the necessary functions to evaluate the model."""
    # Path of the parameters file
    params_path = Path("params.yaml")
    params_path = ROOT_DIR / params_path

    # Read data preparation parameters
    with open(params_path, "r", encoding='utf-8') as params_file:
        try:
            params = yaml.safe_load(params_file)
            params = params["model"]
        except yaml.YAMLError as exc:
            print(exc)

    seed_everything(params["random_state"])
    print("------- Testing of", params["algorithm_name"], "-------")

    # Set Mlflow experiment
    mlflow.set_tracking_uri('https://dagshub.com/armand-07/TAED2-Falsettos.mlflow')
    mlflow.log_param('mode', 'testing')
    mlflow.log_params(params)

    test_loader = data_loading(PROCESSED_DATA_DIR, params["batch_size"])  # data loading

    # ============== #
    # MODEL CREATION #
    # ============== #
    model = HubertForAudioClassification(adapter_hidden_size=params["adapter_hidden_size"])
    # Load the state dictionary and then assign it to the model
    path = os.path.join(MODELS_DIR, f'{params["algorithm_name"]}_bestmodel.pt')
    model.load_state_dict(torch.load(path), strict=False)

    # Define criterion
    criterion = nn.CrossEntropyLoss(reduction='mean')

    # We need GPU to test the model
    assert torch.cuda.is_available()
    print(f'Using CUDA with {torch.cuda.device_count()} GPUs')
    model.cuda()

    # ============== #
    # MODEL TESTING  #
    # ============== #
    print("------------- Testing phase ----------------")

    test_loss, test_f1 = eval_model(test_loader, model, criterion, params["num_classes"])
    mlflow.log_metric("Average Loss Test", test_loss)
    mlflow.log_metric("Best F1-score", test_f1)

    mlflow.pytorch.log_model(model, "model")


main()
