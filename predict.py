"""
predict.py — Run inference with the trained model
====================================================
Usage:
    python predict.py --email "Urgent: verify your account at http://bit.ly/abc"
    python predict.py --file emails.txt
    python predict.py --interactive
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))
from src.model import PhishingEmailDetector

LABEL_MAP = {0: "✅  SAFE", 1: "🚨  PHISHING"}
COLOR_MAP  = {0: "\033[92m", 1: "\033[91m"}
RESET      = "\033[0m"


def format_result(email_preview, label, proba=None):
    preview = email_preview[:80].replace("\n", " ") + ("..." if len(email_preview) > 80 else "")
    color   = COLOR_MAP[label]
    verdict = LABEL_MAP[label]
    line    = f"\n{color}{'─'*60}{RESET}"
    line   += f"\n{color}Email   :{RESET} {preview}"
    line   += f"\n{color}Result  :{RESET} {color}{verdict}{RESET}"
    if proba is not None:
        confidence = proba[label] * 100
        line += f"\n{color}Confidence:{RESET} {confidence:.1f}%"
    return line


def predict_single(model, email_text):
    label = model.predict([email_text])[0]
    try:
        proba = model.predict_proba([email_text])[0]
    except Exception:
        proba = None
    return label, proba


def run_interactive(model):
    print("\n" + "="*60)
    print("  Phishing Email Detector — Interactive Mode")
    print("  Type 'quit' or 'exit' to stop.")
    print("="*60)
    while True:
        print("\nPaste email text (or type 'quit'):")
        lines = []
        while True:
            line = input()
            if line.lower() in ("quit", "exit"):
                print("Goodbye!")
                return
            if line == "":
                break
            lines.append(line)
        if not lines:
            continue
        email_text = "\n".join(lines)
        label, proba = predict_single(model, email_text)
        print(format_result(email_text, label, proba))


def run_file(model, path):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    emails = [e.strip() for e in content.split("\n\n") if e.strip()]
    print(f"\n[*] Predicting {len(emails)} email(s) from {path}\n")
    for i, email in enumerate(emails, 1):
        label, proba = predict_single(model, email)
        print(f"[{i}] {format_result(email, label, proba)}")
    phishing_count = sum(model.predict([e])[0] for e in emails)
    print(f"\n{'='*60}")
    print(f"Summary: {phishing_count}/{len(emails)} emails classified as PHISHING")


def parse_args():
    parser = argparse.ArgumentParser(description="Phishing Email Predictor")
    parser.add_argument("--model",       type=str, default="models/phishing_detector.pkl")
    parser.add_argument("--email",       type=str, default=None)
    parser.add_argument("--file",        type=str, default=None)
    parser.add_argument("--interactive", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.model):
        print(f"[!] Model not found at '{args.model}'. Run train.py first.")
        sys.exit(1)
    model = PhishingEmailDetector.load(args.model)
    if args.email:
        label, proba = predict_single(model, args.email)
        print(format_result(args.email, label, proba))
    elif args.file:
        run_file(model, args.file)
    elif args.interactive:
        run_interactive(model)
    else:
        print("[!] Provide --email, --file, or --interactive. Use --help for usage.")
