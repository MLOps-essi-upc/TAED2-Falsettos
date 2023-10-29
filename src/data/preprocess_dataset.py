"""
Take the raw data and make the preprocess step
"""

import os
from pathlib import Path

import yaml
import torch
import datasets
import numpy as np
import pandas as pd

from transformers import Wav2Vec2FeatureExtractor
from src import ROOT_DIR,RAW_DATA_DIR, PROCESSED_DATA_DIR
from src.features.validate import great_expectations


def preprocess_data(data, feature_extractor, total_labels, audio_length):
    """ Normalize the audio and reencode the labels
    """
    # Audio normalization and feature extraction
    audio_decoded = data["audio"]["array"]
    audio_decoded -= audio_decoded.mean()

    if len(audio_decoded) < audio_length*16000:  # pad if audio is shorter than one second
        audio_decoded = np.pad(audio_decoded, (0, 16000-len(audio_decoded)), 'constant')
    elif len(audio_decoded) > audio_length*16000:  # cut if audio is longer than one second
        audio_decoded = audio_decoded[:16000]

    feature_extractor(audio_decoded, sampling_rate=16000)

    # One hot encoding of the labels
    one_hot_encoding = torch.zeros(total_labels, dtype=torch.float32)
    one_hot_encoding[data['label']] = 1

    return {"key": data['file'], "features": torch.from_numpy
            (audio_decoded.astype(np.float32)), "label": one_hot_encoding}


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    # Path of the parameters file
    params_path = Path("params.yaml")

    # Read data preparation parameters
    with open(params_path, "r", encoding='utf-8') as params_file:
        try:
            params = yaml.safe_load(params_file)
            params = params["dataset"]
        except yaml.YAMLError as exc:
            print(exc)

    print("---------- Dataset preprocessing ------------")

    audio_dataset = datasets.load_from_disk(RAW_DATA_DIR) # Load the dataset
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        params["feature_extractor"]) # Wav2Vec2 feature extractor

    dict_predatasplit = {}  # to store all three splits of the dataset after preprocessing
    df_great_expectations = pd.DataFrame()
    for split, dataset_split in audio_dataset.items():
        df_great_expectations = pd.concat([df_great_expectations, dataset_split.to_pandas()])
        print("Preprocessing the data split: ", split,
              ", with length: ", len(dataset_split), sep = "")
        # to store the preprocessed data for each split
        preprocessed_datasplit = [None]*len(dataset_split)
        for i in enumerate(dataset_split):
            preprocessed_datasplit[i] = preprocess_data(
                dataset_split[i], feature_extractor, params["total_labels"], params["audio_length"]
            )

        print("Creating the dataset for split: ", split, sep = "")
        dict_predatasplit[split] = datasets.Dataset.from_list(preprocessed_datasplit)

    print("------------- Dataset saving ----------------")
    audio_dataset_preprocessed = datasets.DatasetDict(dict_predatasplit)
    print("New preprocessed dataset:", audio_dataset_preprocessed)
    if not os.path.exists(PROCESSED_DATA_DIR):
        os.mkdir(PROCESSED_DATA_DIR)
    audio_dataset_preprocessed.save_to_disk(PROCESSED_DATA_DIR)

    print("------------ Great expectations -------------")
    # reset the index of the dataframe to be the same one for all splits
    df_great_expectations = df_great_expectations.reset_index(drop=True)
    great_expectations(df_great_expectations)
    path_gx = Path(ROOT_DIR, "gx", "uncommitted", "data_docs")
    print("Great expectations done and saved in:", path_gx)


main()
