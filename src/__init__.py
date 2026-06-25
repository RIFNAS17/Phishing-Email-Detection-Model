"""Phishing Email Detection — source package."""
from .model import PhishingEmailDetector, EmailFeatureExtractor
from .dataset import generate_synthetic_dataset, load_csv_dataset

__all__ = [
    "PhishingEmailDetector",
    "EmailFeatureExtractor",
    "generate_synthetic_dataset",
    "load_csv_dataset",
]
