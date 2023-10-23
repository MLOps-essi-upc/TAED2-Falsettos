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

from src import MODELS_DIR, UNKNOWN_WORDS_V2
from src.app.schemas import SpeechCommand, PredictPayload
from src.models.Hubert_Classifier_model import HubertForAudioClassification

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


@app.on_event("startup")
def _load_models():
    """Loads all pickled models found in `MODELS_DIR` and adds them to `models_list`"""

    model_paths = [
        filename
        for filename in MODELS_DIR.iterdir()
        if filename.suffix == ".pt"
    ]

    #for path in model_paths:
    #    with open(path, "rb") as file:
    #        model_wrapper = pickle.load(file)
    #        model_wrappers_list.append(model_wrapper)

    # Path of the parameters file
    params_path = Path("params.yaml")

    # Read data preparation parameters
    with open(params_path, "r") as params_file:
        try:
            params = yaml.safe_load(params_file)
        except yaml.YAMLError as exc:
            print(exc)

    for path in model_paths:
        model = HubertForAudioClassification(adapter_hidden_size=params["model"]["adapter_hidden_size"])
        model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        model_wrappers_list.append(model)
        #model.cpu()
        #model.eval()


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


@app.get("/models", tags=["Prediction"])
@construct_response
def _get_models_list(request: Request, type: str = None):
    """Return the list of available models"""

    available_models = [
        {
            "type": "type",
            "parameters": "muchos",
            "f1 score": "de locos",
        }
        for model in model_wrappers_list
    ]

    #available_models = [
    #    {
    #        "type": model["type"],
    #        "parameters": model["params"],
    #        "f1 score": model["metrics"],
    #    }
    #    for model in model_wrappers_list
    #    if model["type"] == type or type is None
    #]

    if not available_models:
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail="Type not found")
    else:
        return {
            "message": HTTPStatus.OK.phrase,
            "status-code": HTTPStatus.OK,
            "data": available_models,
        }


def _preprocess_data_sample(audio_decoded, feature_extractor, audio_length):
    # Frequency sampling should be 16kHz
    # Audio normalization and feature extraction
    #buffer_size = len(audio_decoded)
    #element_size = np.dtype(np.float64).itemsize
    #padding_size = (buffer_size // element_size) * element_size
    #padded_data = audio_decoded[:padding_size]
    #audio_array = np.frombuffer(padded_data, dtype=np.float64)

    audio_array = audio_decoded - audio_decoded.mean()

    if len(audio_decoded)<audio_length*16000:  # the audio is shorter than one second, we need to pad it
        audio_decoded = np.pad(audio_decoded, (0, 16000-len(audio_decoded)), 'constant')
    elif len(audio_decoded)>audio_length*16000:  # the audio is longer than one second, we need to cut it
        audio_decoded = audio_decoded[:16000]

    feature_extractor(audio_array, sampling_rate = 16000)

    tensor = torch.from_numpy(audio_array.astype(np.float32))
    tensor = tensor.unsqueeze(0) # add batch dimension
    
    return tensor


@app.post("/models/{type}", tags=["Prediction"])
@construct_response
def _predict(request: Request, type: str, payload: PredictPayload):
    """Recognise a Speech Command given an audio file"""

    params = _get_params()
    arr=eval(payload.audio_array)
    data = np.frombuffer(arr)

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(params["dataset"]["feature_extractor"]) # Wav2Vec2 feature extractor
    data_sample = _preprocess_data_sample(data, feature_extractor, params["dataset"]["audio_length"])

    #model_wrapper = next((m for m in model_wrappers_list), None) #if m["type"] == type), None)

    model = model_wrappers_list[0]
    model.eval()
    if True:
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
    else:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST, detail="Model not found"
        )
    return response