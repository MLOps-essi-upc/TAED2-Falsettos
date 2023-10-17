# -*- coding: utf-8 -*-
from src import PROCESSED_DATA_DIR, RAW_DATA_DIR

from pathlib import Path
import yaml
import datasets
import numpy as np
from transformers import Wav2Vec2FeatureExtractor
import torch


def preprocess_data(data, feature_extractor, total_labels, audio_length):
    # Audio normalization and feature extraction
    audio_decoded = data["audio"]["array"]
    audio_decoded -= audio_decoded.mean()

    if len(audio_decoded)<audio_length*16000: #the audio is shorter than one second, we need to pad it
        audio_decoded = np.pad(audio_decoded, (0, 16000-len(audio_decoded)), 'constant')
    elif len(audio_decoded)>audio_length*16000: #the audio is longer than one second, we need to cut it
        audio_decoded = audio_decoded[:16000]

    feature_extractor(audio_decoded, sampling_rate = 16000)

    # One hot encoding of the labels
    one_hot_encoding = torch.zeros(total_labels, dtype = torch.float32)
    one_hot_encoding[data['label']] = 1
    
    return {"key": data['file'], "features": torch.from_numpy(audio_decoded.astype(np.float32)), "label": one_hot_encoding}


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    # Path of the parameters file
    params_path = Path("params.yaml")

    # Read data preparation parameters
    with open(params_path, "r") as params_file:
        try:
            params = yaml.safe_load(params_file)
            params = params["dataset"]
        except yaml.YAMLError as exc:
            print(exc)


    print("---------- Dataset preprocessing ------------")

    audio_dataset = datasets.load_from_disk(RAW_DATA_DIR) # Load the dataset
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(params["feature_extractor"]) # Wav2Vec2 feature extractor

    dict_predatsplit = {} # to store all three splits of the dataset after preprocessing
    for split, dataset_split in audio_dataset.items():
        print("Preprocessing the data split: ", split, ", with length: ", len(dataset_split), sep = "")
        preprocessed_datsplit = [None]*len(dataset_split) # to store the preprocessed data for each split
        for i in range(len(dataset_split)):
            preprocessed_datsplit[i] = preprocess_data(dataset_split[i], feature_extractor, params["total_labels"], params["audio_length"])

        print("Creating the dataset for split: ", split, sep = "")
        dict_predatsplit[split] = datasets.Dataset.from_list(preprocessed_datsplit)

    print("------------- Dataset saving ----------------")
    audio_dataset_preprocessed = datasets.DatasetDict(dict_predatsplit)
    print("New preprocessed dataset:", audio_dataset_preprocessed)
    audio_dataset_preprocessed.save_to_disk(PROCESSED_DATA_DIR)


main()
