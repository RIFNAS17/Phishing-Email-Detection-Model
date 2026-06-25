"""
Phishing Email Detection Model
================================
Uses Scikit-learn to classify emails as Phishing or Safe based on
textual content and URL features.
"""

import os
import re
import pickle
import numpy as np
import pandas as pd
from urllib.parse import urlparse

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix


PHISHING_KEYWORDS = [
    "verify", "account", "suspended", "urgent", "immediately", "login",
    "password", "click here", "confirm", "update", "secure", "bank",
    "paypal", "ebay", "amazon", "apple", "microsoft", "irs", "tax",
    "refund", "prize", "winner", "congratulations", "free", "limited",
    "offer", "expires", "act now", "dear customer", "dear user",
    "validate", "unauthorized", "suspicious", "unusual activity",
    "kindly", "resolve", "failure", "compromised", "locked", "blocked",
    "ssn", "social security", "credit card", "debit", "wire transfer",
    "cryptocurrency", "bitcoin", "inheritance", "lottery", "million",
    "billion", "nigerian", "prince", "beneficiary", "confidential"
]

SUSPICIOUS_DOMAINS = [
    "bit.ly", "tinyurl.com", "goo.gl", "ow.ly", "t.co",
    "is.gd", "buff.ly", "adf.ly", "tiny.cc"
]


class EmailFeatureExtractor:
    """Extracts hand-crafted features from raw email text."""

    def extract(self, email_text: str) -> dict:
        text_lower = email_text.lower()
        urls = self._extract_urls(email_text)
        return {
            "num_keywords":        self._count_keywords(text_lower),
            "exclamation_count":   email_text.count("!"),
            "question_count":      email_text.count("?"),
            "uppercase_ratio":     self._uppercase_ratio(email_text),
            "text_length":         len(email_text),
            "word_count":          len(email_text.split()),
            "has_html":            int(bool(re.search(r"<[a-z][\s\S]*>", email_text, re.I))),
            "num_links":           len(urls),
            "has_ip_url":          int(any(self._is_ip_url(u) for u in urls)),
            "has_shortened_url":   int(any(self._is_shortened(u) for u in urls)),
            "avg_url_length":      np.mean([len(u) for u in urls]) if urls else 0,
            "max_url_length":      max((len(u) for u in urls), default=0),
            "num_suspicious_domains": sum(self._is_shortened(u) for u in urls),
            "has_unsubscribe":     int("unsubscribe" in text_lower),
            "has_dear":            int(bool(re.search(r"\bdear\b", text_lower))),
            "digit_ratio":         self._digit_ratio(email_text),
        }

    def _extract_urls(self, text):
        return re.findall(r'https?://[^\s<>"{}|\\^`\[\]]+', text)

    def _count_keywords(self, text):
        return sum(kw in text for kw in PHISHING_KEYWORDS)

    def _uppercase_ratio(self, text):
        letters = [c for c in text if c.isalpha()]
        return sum(c.isupper() for c in letters) / len(letters) if letters else 0

    def _digit_ratio(self, text):
        chars = [c for c in text if c.isalnum()]
        return sum(c.isdigit() for c in chars) / len(chars) if chars else 0

    def _is_ip_url(self, url):
        try:
            host = urlparse(url).hostname or ""
            return bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", host))
        except Exception:
            return False

    def _is_shortened(self, url):
        try:
            host = urlparse(url).hostname or ""
            return any(host == d or host.endswith("." + d) for d in SUSPICIOUS_DOMAINS)
        except Exception:
            return False

    def fit_transform(self, emails):
        return np.array([list(self.extract(e).values()) for e in emails])

    def transform(self, emails):
        return self.fit_transform(emails)

    @property
    def feature_names(self):
        return list(self.extract("sample").keys())


class PhishingEmailDetector:
    """End-to-end phishing email detection pipeline."""

    CLASSIFIERS = {
        "random_forest":        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "gradient_boosting":    GradientBoostingClassifier(n_estimators=150, random_state=42),
        "logistic_regression":  LogisticRegression(max_iter=1000, random_state=42),
        "svm":                  LinearSVC(max_iter=2000, random_state=42),
    }

    def __init__(self, classifier: str = "random_forest"):
        if classifier not in self.CLASSIFIERS:
            raise ValueError(f"classifier must be one of {list(self.CLASSIFIERS)}")
        self.clf_name = classifier
        self.classifier = self.CLASSIFIERS[classifier]
        self.tfidf = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True
        )
        self.feature_extractor = EmailFeatureExtractor()
        self.scaler = StandardScaler(with_mean=False)
        self._trained = False

    def fit(self, emails, labels):
        X_tfidf = self.tfidf.fit_transform(emails)
        X_hand  = csr_matrix(self.feature_extractor.fit_transform(emails))
        X_hand  = self.scaler.fit_transform(X_hand)
        X       = hstack([X_tfidf, X_hand])
        self.classifier.fit(X, labels)
        self._trained = True
        return self

    def predict(self, emails):
        self._check_trained()
        X = self._build_features(emails)
        return self.classifier.predict(X)

    def predict_proba(self, emails):
        self._check_trained()
        if not hasattr(self.classifier, "predict_proba"):
            raise AttributeError(f"{self.clf_name} does not support predict_proba.")
        X = self._build_features(emails)
        return self.classifier.predict_proba(X)

    def evaluate(self, emails, labels):
        preds  = self.predict(emails)
        acc    = accuracy_score(labels, preds)
        report = classification_report(labels, preds, target_names=["Safe", "Phishing"])
        cm     = confusion_matrix(labels, preds)
        return {"accuracy": acc, "report": report, "confusion_matrix": cm}

    def save(self, path: str = "models/phishing_detector.pkl"):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"[✓] Model saved → {path}")

    @classmethod
    def load(cls, path: str = "models/phishing_detector.pkl"):
        with open(path, "rb") as f:
            model = pickle.load(f)
        print(f"[✓] Model loaded ← {path}")
        return model

    def _build_features(self, emails):
        X_tfidf = self.tfidf.transform(emails)
        X_hand  = csr_matrix(self.feature_extractor.transform(emails))
        X_hand  = self.scaler.transform(X_hand)
        return hstack([X_tfidf, X_hand])

    def _check_trained(self):
        if not self._trained:
            raise RuntimeError("Model not trained. Call fit() first.")
