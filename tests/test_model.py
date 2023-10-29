"""
This module contains the model testing.
"""

import pytest

from src.models import predict_model


def test_model():
    """Test the model prediction."""
    expected_label = 20
    predicted_label = predict_model.predict_label()
    assert expected_label == predicted_label
