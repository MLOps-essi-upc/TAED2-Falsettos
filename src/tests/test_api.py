import pytest
from http import HTTPStatus
from fastapi.testclient import TestClient
from src.app.api import app

from src import ROOT_DIR
import os
import pandas as pd


@pytest.fixture
def client():
    with TestClient(app) as client:
        return client


@pytest.fixture
def payload():
    RAW_DATA_SAMPLE = os.path.join(ROOT_DIR, 'data', 'raw_sample_example', 'sample_example.pkl')
    sample_df = pd.read_pickle(RAW_DATA_SAMPLE)
    array_string = str(sample_df["audio_array"][0].tobytes())
    return array_string


def test_api_init(client):
    response = client.get("/")
    json = response.json()
    assert response.status_code == 200
    assert (
        json["data"]["message"] == "Welcome to Speech Command Recognizer! Please, read the `/docs`!"
    )
    assert json["message"] == "OK"
    assert json["status-code"] == 200


def test_api_model(client, payload):
    label = 20
    predicted_command = "bed"
    response = client.post("/predict", payload)
    assert response.status_code == 200
    assert response.json() == {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {
            "predicted_label": label,
            "predicted_command": predicted_command,
        },
    }