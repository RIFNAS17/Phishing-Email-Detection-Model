"""
train.py — Train the Phishing Email Detection Model
======================================================
Usage:
    python train.py                          # synthetic data, random_forest
    python train.py --data data/emails.csv   # real CSV
    python train.py --clf gradient_boosting  # choose classifier
    python train.py --compare                # compare all classifiers
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from src.model import PhishingEmailDetector
from src.dataset import generate_synthetic_dataset, load_csv_dataset


def plot_confusion_matrix(cm, save_path="confusion_matrix.png"):
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Safe", "Phishing"],
        yticklabels=["Safe", "Phishing"],
        linewidths=0.5
    )
    plt.title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.ylabel("Actual Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[✓] Confusion matrix saved → {save_path}")


def plot_feature_importance(model, top_n=20, save_path="feature_importance.png"):
    clf = model.classifier
    if not hasattr(clf, "feature_importances_"):
        print("[!] Feature importance not available for this classifier.")
        return
    importances = clf.feature_importances_
    n_tfidf     = len(model.tfidf.get_feature_names_out())
    hand_names  = model.feature_extractor.feature_names
    hand_imp = importances[n_tfidf:]
    indices  = np.argsort(hand_imp)[::-1][:top_n]
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(indices)), hand_imp[indices], color="steelblue")
    plt.xticks(range(len(indices)), [hand_names[i] for i in indices], rotation=45, ha="right")
    plt.title("Top Hand-Crafted Feature Importances", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[✓] Feature importance plot saved → {save_path}")


def plot_classifier_comparison(results, save_path="classifier_comparison.png"):
    names  = list(results.keys())
    scores = [results[n]["cv_mean"] for n in names]
    errors = [results[n]["cv_std"]  for n in names]
    plt.figure(figsize=(9, 5))
    bars = plt.bar(names, scores, yerr=errors, capsize=5, color="steelblue", alpha=0.85)
    plt.ylim(0.5, 1.05)
    plt.ylabel("Cross-Val Accuracy", fontsize=12)
    plt.title("Classifier Comparison (5-Fold CV)", fontsize=14, fontweight="bold")
    for bar, s in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{s:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[✓] Classifier comparison saved → {save_path}")


def train_and_evaluate(df, clf_name="random_forest", output_dir="models"):
    os.makedirs(output_dir, exist_ok=True)
    emails = df["email_text"].tolist()
    labels = df["label"].tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        emails, labels, test_size=0.2, stratify=labels, random_state=42
    )
    print(f"\n{'='*55}")
    print(f"  Classifier : {clf_name}")
    print(f"  Train size : {len(X_train)}")
    print(f"  Test size  : {len(X_test)}")
    print(f"{'='*55}")
    model = PhishingEmailDetector(classifier=clf_name)
    model.fit(X_train, y_train)
    results = model.evaluate(X_test, y_test)
    print(f"\n  Accuracy   : {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    print(f"\n{results['report']}")
    plot_confusion_matrix(results["confusion_matrix"],
                          save_path=os.path.join(output_dir, "confusion_matrix.png"))
    plot_feature_importance(model,
                            save_path=os.path.join(output_dir, "feature_importance.png"))
    model.save(os.path.join(output_dir, "phishing_detector.pkl"))
    return model, results


def compare_classifiers(df, output_dir="models"):
    from sklearn.model_selection import StratifiedKFold
    emails = np.array(df["email_text"].tolist())
    labels = np.array(df["label"].tolist())
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    comparison = {}
    for clf_name in PhishingEmailDetector.CLASSIFIERS:
        print(f"\n[*] Evaluating: {clf_name} ...")
        fold_scores = []
        for train_idx, val_idx in cv.split(emails, labels):
            m = PhishingEmailDetector(classifier=clf_name)
            m.fit(emails[train_idx].tolist(), labels[train_idx].tolist())
            r = m.evaluate(emails[val_idx].tolist(), labels[val_idx].tolist())
            fold_scores.append(r["accuracy"])
        comparison[clf_name] = {
            "cv_mean": np.mean(fold_scores),
            "cv_std":  np.std(fold_scores),
        }
        print(f"    CV Accuracy: {comparison[clf_name]['cv_mean']:.4f} ± {comparison[clf_name]['cv_std']:.4f}")
    os.makedirs(output_dir, exist_ok=True)
    plot_classifier_comparison(comparison,
                               save_path=os.path.join(output_dir, "classifier_comparison.png"))
    best = max(comparison, key=lambda k: comparison[k]["cv_mean"])
    print(f"\n[★] Best classifier: {best}")
    return comparison, best


def parse_args():
    parser = argparse.ArgumentParser(description="Train Phishing Email Detector")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--clf",  type=str, default="random_forest",
                        choices=list(PhishingEmailDetector.CLASSIFIERS.keys()))
    parser.add_argument("--n-phishing", type=int, default=1000)
    parser.add_argument("--n-safe",     type=int, default=1000)
    parser.add_argument("--output",     type=str, default="models")
    parser.add_argument("--compare",    action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.data:
        df = load_csv_dataset(args.data)
    else:
        print("[*] No dataset provided — generating synthetic data...")
        df = generate_synthetic_dataset(args.n_phishing, args.n_safe)
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/emails_dataset.csv", index=False)
    if args.compare:
        comparison, best_clf = compare_classifiers(df, output_dir=args.output)
        train_and_evaluate(df, clf_name=best_clf, output_dir=args.output)
    else:
        train_and_evaluate(df, clf_name=args.clf, output_dir=args.output)
