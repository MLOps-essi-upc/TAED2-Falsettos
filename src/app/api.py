"""Main script: it includes our API initialization and endpoints."""

from pathlib import Path
import yaml

import pickle
from datetime import datetime
from functools import wraps
from http import HTTPStatus
from typing import List

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from typing import Annotated
from transformers import Wav2Vec2FeatureExtractor
import torch.nn.functional as F

from src import MODELS_DIR
from src.app.schemas import SpeechCommand, PredictPayload
from src.models.Hubert_Classifier_model import HubertForAudioClassification
import os

import soundfile as sf

model_wrappers_list: List[dict] = []

# Define application
app = FastAPI(
    title="Speech Command Recognition",
    description="This API lets you make predictions on the SpeechCommands dataset using the HubertModel.",
    version="0.1",
)


def construct_response(f):
    """Construct a JSON response for an endpoint's results."""

    @wraps(f)
    def wrap(request: Request, *args, **kwargs):
        results = f(request, *args, **kwargs)

        # Construct response
        response = {
            "message": results["message"],
            "method": request.method,
            "status-code": results["status-code"],
            "timestamp": datetime.now().isoformat(),
            "url": request.url._url,
        }

        # Add data
        if "data" in results:
            response["data"] = results["data"]

        return response

    return wrap


def _get_params():
    """Read data preparation parameters"""

    params_path = Path("params.yaml")
    with open(params_path, "r") as params_file:
        try:
            params = yaml.safe_load(params_file)
        except yaml.YAMLError as exc:
            print(exc)
    return params


@app.on_event("startup")
def _load_model():
    """Load the HubertModel for Audio Classification"""

    model_path = os.path.join(MODELS_DIR,"Hubert_Classifier_bestmodel.pt")
    params = _get_params()

    model = HubertForAudioClassification(adapter_hidden_size=params["model"]["adapter_hidden_size"])
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    return model


@app.get("/", tags=["General"])  # path operation decorator
@construct_response
def _index(request: Request):
    """Root endpoint."""

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"message": "Welcome to Speech Command Recognizer! Please, read the `/docs`!"},
    }
    return response


def _preprocess_data_sample(audio_decoded, feature_extractor, audio_length):
    # Frequency sampling should be 16kHz
    # Audio normalization and feature extraction
    #buffer_size = len(audio_decoded)
    #element_size = np.dtype(np.float64).itemsize
    #padding_size = (buffer_size // element_size) * element_size
    #padded_data = audio_decoded[:padding_size]
    #audio_array = np.frombuffer(padded_data, dtype=np.float64)

    audio_array = audio_decoded-audio_decoded.mean()

    if len(audio_decoded)<audio_length*16000:  # the audio is shorter than one second, we need to pad it
        audio_decoded = np.pad(audio_decoded, (0, 16000-len(audio_decoded)), 'constant')
    elif len(audio_decoded)>audio_length*16000:  # the audio is longer than one second, we need to cut it
        audio_decoded = audio_decoded[:16000]

    feature_extractor(audio_array, sampling_rate = 16000)

    tensor = torch.from_numpy(audio_array.astype(np.float32))
    tensor = tensor.unsqueeze(0) # add batch dimension

    return tensor


@app.post("/predict", tags=["Prediction"])
@construct_response
def _predict(request: Request, payload: PredictPayload):
    """Recognise a Speech Command given an audio file"""

    params = _get_params()
    data = np.frombuffer(eval(payload.audio_array))

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(params["dataset"]["feature_extractor"]) # Wav2Vec2 feature extractor
    data_sample = _preprocess_data_sample(data, feature_extractor, params["dataset"]["audio_length"])

    model = _load_model()
    model.eval()
    with torch.no_grad():
    # Get the output
        logits = model(data_sample)
        output = F.log_softmax(logits, dim=1)
        # Get the indices of the maximum values along the class dimension
        predicted_class = torch.argmax(output, dim=1)

    label = predicted_class.int().item()
    predicted_command = SpeechCommand(label).name

    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "predicted_label": label,
            "predicted_command": predicted_command,
        },
    }
    return response
