"""
This module contains the api testing.
"""

import os
from http import HTTPStatus

import pytest
import pandas as pd
from fastapi.testclient import TestClient

from src import ROOT_DIR
from src.app.api import app


@pytest.fixture(scope="module", autouse=True)
def client():
    """Define client object for running pytest."""
    with TestClient(app) as client:
        return client


@pytest.fixture
def payload():
    """Define payload object for running pytest."""
    raw_data_sample = os.path.join(ROOT_DIR, 'data', 'raw_sample_example', 'sample_example.pkl')
    sample_df = pd.read_pickle(raw_data_sample)
    array_string = str(sample_df["audio_array"][0].tobytes())
    return {'audio_array': array_string}


def test_api_init(client):
    """Test the api initialization."""
    response = client.get("/")
    json = response.json()
    assert response.status_code == 200
    assert (
        json["data"]["message"] == "Welcome to Speech Command Recognizer! Please, read the `/docs`!"
    )
    assert json["message"] == "OK"
    assert json["status-code"] == 200


def test_api_model(client, payload):
    """Test the api model prediction."""
    label = 20
    predicted_command = "Bed"
    response = client.post("/predict", json=payload)
    json_response = response.json()
    assert response.status_code == 200
    assert json_response["message"] == HTTPStatus.OK.phrase
    assert json_response["status-code"] == HTTPStatus.OK
    assert json_response["data"]["predicted_label"] == label
    assert json_response["data"]["predicted_command"] == predicted_command
