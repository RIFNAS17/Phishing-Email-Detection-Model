"""
tests/test_model.py — Unit tests for the Phishing Email Detector
"""

import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.model import PhishingEmailDetector, EmailFeatureExtractor
from src.dataset import generate_synthetic_dataset


PHISHING_SAMPLES = [
    "Dear Customer, your account is suspended. Verify at http://bit.ly/abc123 immediately!",
    "URGENT: Unusual activity detected. Click here to confirm: http://192.168.1.1/login",
    "Congratulations! You won $5000! Claim now at https://tinyurl.com/xyz",
]

SAFE_SAMPLES = [
    "Hi Alice, the team meeting is at 2pm tomorrow. Please review the agenda.",
    "Your order #123456 has shipped. Expected delivery: Monday, July 5.",
    "Thank you for contacting support. Your ticket #789 has been resolved.",
]


@pytest.fixture(scope="module")
def trained_model():
    df = generate_synthetic_dataset(300, 300)
    model = PhishingEmailDetector(classifier="random_forest")
    model.fit(df["email_text"].tolist(), df["label"].tolist())
    return model


class TestEmailFeatureExtractor:

    def test_returns_dict(self):
        fe = EmailFeatureExtractor()
        result = fe.extract("sample email text")
        assert isinstance(result, dict)

    def test_correct_keys(self):
        fe = EmailFeatureExtractor()
        result = fe.extract("test")
        expected = [
            "num_keywords", "exclamation_count", "question_count",
            "uppercase_ratio", "text_length", "word_count", "has_html",
            "num_links", "has_ip_url", "has_shortened_url", "avg_url_length",
            "max_url_length", "num_suspicious_domains", "has_unsubscribe",
            "has_dear", "digit_ratio"
        ]
        for k in expected:
            assert k in result, f"Missing key: {k}"

    def test_detects_ip_url(self):
        fe = EmailFeatureExtractor()
        result = fe.extract("Click here: http://192.168.1.1/login")
        assert result["has_ip_url"] == 1

    def test_detects_shortened_url(self):
        fe = EmailFeatureExtractor()
        result = fe.extract("Go to http://bit.ly/abc")
        assert result["has_shortened_url"] == 1

    def test_counts_exclamation(self):
        fe = EmailFeatureExtractor()
        result = fe.extract("Urgent!!! Act now!!!")
        assert result["exclamation_count"] == 6

    def test_fit_transform_shape(self):
        fe = EmailFeatureExtractor()
        X = fe.fit_transform(["email one", "email two", "email three"])
        assert X.shape[0] == 3


class TestPhishingEmailDetector:

    def test_invalid_classifier_raises(self):
        with pytest.raises(ValueError):
            PhishingEmailDetector(classifier="invalid_clf")

    def test_predict_before_fit_raises(self):
        model = PhishingEmailDetector()
        with pytest.raises(RuntimeError):
            model.predict(["test email"])

    def test_fit_and_predict(self, trained_model):
        preds = trained_model.predict(PHISHING_SAMPLES + SAFE_SAMPLES)
        assert len(preds) == 6
        assert all(p in [0, 1] for p in preds)

    def test_phishing_detection(self, trained_model):
        preds = trained_model.predict(PHISHING_SAMPLES)
        assert sum(preds) >= 2

    def test_safe_detection(self, trained_model):
        preds = trained_model.predict(SAFE_SAMPLES)
        assert sum(p == 0 for p in preds) >= 2

    def test_evaluate_returns_keys(self, trained_model):
        df = generate_synthetic_dataset(50, 50)
        result = trained_model.evaluate(
            df["email_text"].tolist(), df["label"].tolist()
        )
        assert "accuracy" in result
        assert "report" in result
        assert "confusion_matrix" in result

    def test_accuracy_above_threshold(self, trained_model):
        df = generate_synthetic_dataset(200, 200)
        result = trained_model.evaluate(
            df["email_text"].tolist(), df["label"].tolist()
        )
        assert result["accuracy"] >= 0.80, f"Accuracy too low: {result['accuracy']}"

    def test_save_and_load(self, trained_model, tmp_path):
        save_path = str(tmp_path / "test_model.pkl")
        trained_model.save(save_path)
        loaded = PhishingEmailDetector.load(save_path)
        preds_orig   = trained_model.predict(SAFE_SAMPLES)
        preds_loaded = loaded.predict(SAFE_SAMPLES)
        assert list(preds_orig) == list(preds_loaded)

    def test_predict_proba(self, trained_model):
        proba = trained_model.predict_proba(PHISHING_SAMPLES[:2])
        assert proba.shape == (2, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_all_classifiers(self):
        df = generate_synthetic_dataset(100, 100)
        for clf_name in PhishingEmailDetector.CLASSIFIERS:
            model = PhishingEmailDetector(classifier=clf_name)
            model.fit(df["email_text"].tolist(), df["label"].tolist())
            preds = model.predict(PHISHING_SAMPLES)
            assert len(preds) == len(PHISHING_SAMPLES), f"Failed for {clf_name}"


class TestDatasetGenerator:

    def test_generates_correct_count(self):
        df = generate_synthetic_dataset(50, 50)
        assert len(df) == 100

    def test_balanced_classes(self):
        df = generate_synthetic_dataset(100, 100)
        assert df["label"].sum() == 100
        assert (df["label"] == 0).sum() == 100

    def test_columns_present(self):
        df = generate_synthetic_dataset(10, 10)
        assert "email_text" in df.columns
        assert "label" in df.columns

    def test_no_null_values(self):
        df = generate_synthetic_dataset(50, 50)
        assert df["email_text"].isnull().sum() == 0
        assert df["label"].isnull().sum() == 0
