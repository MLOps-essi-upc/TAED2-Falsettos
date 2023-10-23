"""Definitions for the objects used by our resource endpoints."""

from enum import Enum
from pydantic import BaseModel
import numpy as np
import numpy

from pydantic import BaseModel




class PredictPayload(BaseModel):
    audio_array: str


class SpeechCommand(Enum):
    Yes = 0
    No = 1
    Up = 2
    Down = 3
    Left = 4
    Right = 5
    On = 6
    Off = 7
    Stop = 8
    Go = 9
    Zero = 10
    One = 11
    Two = 12
    Three = 13
    Four = 14
    Five = 15
    Six = 16
    Seven = 17
    Eight = 18
    Nine = 19
    Bed = 20
    Bird = 21
    Cat = 22
    Dog = 23
    Happy = 24
    House = 25
    Marvin = 26
    Sheila = 27
    Tree = 28
    Wow = 29
    Backward = 30
    Forward = 31
    Follow = 32
    Learn = 33
    Visual = 34