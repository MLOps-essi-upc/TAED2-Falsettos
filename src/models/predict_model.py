""" Given an audio file, predicts its label
"""
import os
from pathlib import Path

import yaml

import numpy as np
import pandas as pd

from transformers import Wav2Vec2FeatureExtractor

import torch
import torch.nn.functional as F

from src import MODELS_DIR, RAW_DATA_SAMPLE, UNKNOWN_WORDS_V2
from src.models.Hubert_Classifier_model import HubertForAudioClassification


def preprocess_data_sample(audio_decoded, feature_extractor, audio_length):
    """ Takes the file and preprocesses
    """
    # Frequency sampling should be 16kHz
    # Audio normalization and feature extraction
    audio_decoded -= audio_decoded.mean()

    # the audio is shorter than one second, we need to pad it
    if len(audio_decoded)<audio_length*16000:
        audio_decoded = np.pad(audio_decoded, (0, 16000-len(audio_decoded)), 'constant')
    # the audio is longer than one second, we need to cut it
    elif len(audio_decoded)>audio_length*16000:
        audio_decoded = audio_decoded[:16000]

    feature_extractor(audio_decoded, sampling_rate = 16000)

    tensor = torch.from_numpy(audio_decoded.astype(np.float32))
    tensor = tensor.unsqueeze(0) # add batch dimension
    return tensor


def predict_label():
    """ Given preprocessed data predicts its label
    """
    # Path of the parameters file
    params_path = Path("params.yaml")

    # Read data preparation parameters
    with open(params_path, "r") as params_file:
        try:
            params = yaml.safe_load(params_file)
        except yaml.YAMLError as exc:
            print(exc)

    # ============== #
    # Data sample    #
    # ============== #
    # Load the data sample 0 from the pickle file
    sample_df = pd.read_pickle(os.path.join(RAW_DATA_SAMPLE, 'sample_example.pkl'))
    print(sample_df)

    # Wav2Vec2 feature extractor
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        params["dataset"]["feature_extractor"])
    data_sample = preprocess_data_sample(sample_df['audio_array'][0],
                    feature_extractor, params["dataset"]["audio_length"])

    # ============== #
    # MODEL CREATION #
    # ============== #
    model = HubertForAudioClassification(adapter_hidden_size=128) # 128 is the best_model value
    # Load the state dictionary and then assign it to the model
    PATH = os.path.join(MODELS_DIR, '{}_bestmodel.pt'
                        .format(params["model"]["algorithm_name"]))
    model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))

    # Move the model to CPU
    model.cpu()
    model.eval()
    with torch.no_grad():
        # Get the output
        logits = model(data_sample)
        output = F.log_softmax(logits, dim=1)
        # Get the indices of the maximum values along the class dimension
        predicted_class = torch.argmax(output, dim=1)

    print("Log softmax:", torch.squeeze(output).numpy())
    label = predicted_class.int().item()
    print("Predicted label:", label)
    print("Predicted word:", UNKNOWN_WORDS_V2[label])

    return label
