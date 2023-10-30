"""
Define global variables.
"""

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(Path(__file__).resolve().parent.parent)

RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
RAW_DATA_SAMPLE = ROOT_DIR / "data" / "raw_sample_example"

METRICS_DIR = ROOT_DIR / "metrics"
MODELS_DIR = ROOT_DIR / "models" / "final_model"


UNKNOWN_WORDS_V1 = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "bed",
    "bird",
    "cat",
    "dog",
    "happy",
    "house",
    "marvin",
    "sheila",
    "tree",
    "wow",
]

UNKNOWN_WORDS_V2 = UNKNOWN_WORDS_V1 + [
    "backward",
    "forward",
    "follow",
    "learn",
    "visual",
]
