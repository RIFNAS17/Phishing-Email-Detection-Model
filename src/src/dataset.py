"""
Dataset Generator
=================
Generates a realistic synthetic dataset of phishing and safe emails.
Also supports loading real CSV datasets.
"""

import os
import random
import pandas as pd
import numpy as np

random.seed(42)
np.random.seed(42)

PHISHING_TEMPLATES = [
    "Dear Customer, Your {bank} account has been suspended. Click here to verify: http://{ip}/login?id={rand}",
    "URGENT: Your PayPal account shows unusual activity! Verify immediately at http://bit.ly/{rand} or your account will be locked.",
    "Congratulations! You have won a ${amount} prize. Claim now: https://tinyurl.com/{rand}. Act now - expires in 24 hours!",
    "Dear user, We detected unauthorized access to your account. Login immediately to confirm: http://{ip}/secure/{rand}",
    "IRS NOTICE: You are eligible for a tax refund of ${amount}. Update your details: http://{domain}-secure.com/irs/{rand}",
    "Your Apple ID has been compromised. Validate your account now: http://{domain}.verify-apple.com/id={rand}",
    "WINNER! Microsoft has selected you! Claim your FREE gift card. Click: https://bit.ly/{rand} LIMITED TIME!",
    "Dear Account Holder, Your {bank} debit card has been blocked. Resolve: http://{ip}/unblock?token={rand}",
    "Dear {name}, I am a prince from Nigeria with $4.5 billion. Please provide your bank details to receive your share.",
    "Amazon Security Alert: Unusual login detected. Verify now or your account will be terminated: http://amaz0n-{rand}.com",
    "Your password will expire in 24 hours! Update immediately: http://{domain}-login.net/reset/{rand}",
    "Credit card ending {digits} was used at {store}. Not you? Click: http://bit.ly/{rand} to dispute transaction.",
    "Kindly confirm your social security number at http://{ip}/ssn-verify?id={rand} to avoid account suspension.",
    "FINAL NOTICE: Your email account will be closed. Click to keep it active: https://tinyurl.com/{rand}",
    "You have a pending cryptocurrency transfer of 2.3 BTC. Confirm at http://{ip}/crypto/{rand}",
]

SAFE_TEMPLATES = [
    "Hi {name}, just a reminder that your team meeting is scheduled for tomorrow at 2pm. Please review the agenda attached.",
    "Your order #{order} has been shipped! Track your package at our official website. Expected delivery: {date}.",
    "Hello, the monthly newsletter is ready. This month we cover industry trends and upcoming product updates.",
    "Thank you for your recent purchase. Your receipt is attached. Contact support@company.com if you have questions.",
    "Hi {name}, welcome to our platform! Here are some tips to get you started with your new account.",
    "Project update: The development team has completed sprint 4. Please review the attached status report.",
    "Your subscription renewal is coming up on {date}. Log into your account to manage your preferences.",
    "Meeting notes from {date}: Discussed Q3 roadmap, assigned tasks, and reviewed customer feedback summary.",
    "Reminder: Please complete your annual performance review by {date}. Log into the HR portal to get started.",
    "Hi team, the weekly standup is at 9am Monday. Agenda items: deployment status, bug backlog, and Q4 planning.",
    "Your account statement for {month} is now available. Sign in to view your transaction history and balance.",
    "We wanted to let you know that we've updated our privacy policy. No action is required on your part.",
    "Hi {name}, your support ticket #{order} has been resolved. Let us know if you need further assistance.",
    "Lunch and Learn this Friday: 'Introduction to Machine Learning' — Room 3B at noon. All are welcome!",
    "The quarterly report has been published. Please find it in the shared drive under Finance > Reports > Q3.",
]


def _rand_str(n=6):
    return "".join(random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=n))

def _rand_ip():
    return ".".join(str(random.randint(1, 255)) for _ in range(4))

def _rand_domain():
    return random.choice(["secure", "login", "update", "verify", "account"])

def _fill_phishing(template):
    return template.format(
        bank=random.choice(["Chase", "Wells Fargo", "Bank of America", "Citibank"]),
        ip=_rand_ip(),
        rand=_rand_str(),
        amount=random.randint(500, 5000),
        domain=_rand_domain(),
        name=random.choice(["Customer", "User", "Sir/Madam"]),
        digits=random.randint(1000, 9999),
        store=random.choice(["Walmart", "Target", "Best Buy"]),
    )

def _fill_safe(template):
    from datetime import date, timedelta
    future = date.today() + timedelta(days=random.randint(1, 30))
    return template.format(
        name=random.choice(["Alice", "Bob", "Carol", "David", "Emma"]),
        order=random.randint(100000, 999999),
        date=future.strftime("%B %d, %Y"),
        month=future.strftime("%B %Y"),
    )


def generate_synthetic_dataset(n_phishing: int = 500, n_safe: int = 500) -> pd.DataFrame:
    phishing_emails = [_fill_phishing(random.choice(PHISHING_TEMPLATES)) for _ in range(n_phishing)]
    safe_emails     = [_fill_safe(random.choice(SAFE_TEMPLATES))         for _ in range(n_safe)]
    emails = phishing_emails + safe_emails
    labels = [1] * n_phishing + [0] * n_safe
    combined = list(zip(emails, labels))
    random.shuffle(combined)
    emails, labels = zip(*combined)
    df = pd.DataFrame({"email_text": emails, "label": labels})
    print(f"[✓] Generated dataset: {n_phishing} phishing + {n_safe} safe = {len(df)} total")
    return df


def load_csv_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["text", "body", "email", "message", "content"]:
        if col in df.columns and "email_text" not in df.columns:
            df = df.rename(columns={col: "email_text"})
            break
    for col in ["spam", "phishing", "class", "category", "target"]:
        if col in df.columns and "label" not in df.columns:
            df = df.rename(columns={col: "label"})
            break
    assert "email_text" in df.columns, "Dataset must have an email text column"
    assert "label" in df.columns,      "Dataset must have a label column"
    df["label"] = df["label"].astype(int)
    df = df.dropna(subset=["email_text", "label"])
    print(f"[✓] Loaded dataset: {len(df)} emails ({df['label'].sum()} phishing)")
    return df


if __name__ == "__main__":
    df = generate_synthetic_dataset(1000, 1000)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/emails_dataset.csv", index=False)
    print(df.head())
