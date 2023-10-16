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

from pathlib import Path

from mlflow import log_params, log_param, log_metric, set_tracking_uri, start_run, end_run


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


class HubertForAudioClassification(nn.Module):
    def __init__(self, adapter_hidden_size = 64):
        super().__init__()

        self.hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")

        hidden_size = self.hubert.config.hidden_size

        self.adaptor = nn.Sequential(
            nn.Linear(hidden_size, adapter_hidden_size),
            nn.ReLU(True),
            nn.Dropout(0.05),
            nn.Linear(adapter_hidden_size, hidden_size),
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, adapter_hidden_size),
            nn.ReLU(True),
            nn.Dropout(0.05),
            nn.Linear(adapter_hidden_size, 36), # 36 classes
        )

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.hubert.feature_extractor._freeze_parameters()

    def forward(self, x):
        # x shape: (B,E)
        x = self.hubert(x).last_hidden_state
        x = F.layer_norm(x, x.shape[1:])
        x = self.adaptor(x)
        # pooling
        x, _ = x.max(dim=1)

        # Mutilayer perceptron with log softmax for classification
        out = self.classifier(x)

        # Remove last dimension
        return out
        # return shape: (B, total_labels)


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
            params = params["train"]
        except yaml.YAMLError as exc:
            print(exc)


    seed_everything(params["random_state"])
    print("------- Testing of",params["algorithm_name"],"-------")

    # Set Mlflow experiment
    start_run()
    set_tracking_uri('https://dagshub.com/armand-07/TAED2-Falsettos.mlflow')
    log_param('mode', 'testing')
    log_params(params)

    test_loader = data_loading (PROCESSED_DATA_DIR, params["batch_size"]) # data loading

    # ============== #
    # MODEL CREATION #
    # ============== #
    model = HubertForAudioClassification(adapter_hidden_size = params["adapter_hidden_size"])
    model = model.load_state_dict(torch.load(os.path.join(ROOT_DIR, 'models', '{}_bestmodel.pt'.format(params["algorithm_name"]))))
    
   

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
    log_metric("Average Loss Test", test_loss)
    log_metric("Best F1-score", test_F1)

    print(f'F1-Score: {test_F1*100:.1f}')
    end_run()


main()