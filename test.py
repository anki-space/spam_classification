import joblib
import numpy as np
import re
import pandas as pd

# --- Feature Extraction Function (from test_model.py) ---
def extract_features_from_email(email_text, feature_names):

    features = {name: 0 for name in feature_names}
    word_freq_features = [name.split('_')[-1] for name in feature_names if name.startswith('word_freq_')]
    words = re.findall(r'\b\w+\b', email_text.lower())
    num_words = len(words)

    if num_words > 0:
        for word in word_freq_features:
            features[f'word_freq_{word}'] = (words.count(word) / num_words) * 100

    char_freq_features = [name.split('_')[-1] for name in feature_names if name.startswith('char_freq_')]
    num_chars = len(email_text)

    if num_chars > 0:
        for char in char_freq_features:
            features[f'char_freq_{char}'] = (email_text.count(char) / num_chars) * 100

    capital_runs = re.findall(r'[A-Z]+', email_text)
    num_capital_runs = len(capital_runs)

    if num_capital_runs > 0:
        run_lengths = [len(run) for run in capital_runs]
        features['capital_run_length_average'] = sum(run_lengths) / num_capital_runs
        features['capital_run_length_longest'] = max(run_lengths)
        features['capital_run_length_total'] = sum(run_lengths)
    else:
        features['capital_run_length_average'] = 1
        features['capital_run_length_longest'] = 1
        features['capital_run_length_total'] = 0

    return np.array([features[name] for name in feature_names])
# --- Classification Function ---
def classify_email(email_text):
    try:
        model = joblib.load('models/spam_classifier_rf.joblib')
        scaler = joblib.load('models/scaler.joblib')
        feature_names = joblib.load('models/feature_names.joblib')
    except FileNotFoundError:
        return "Error: Model or scaler files not found. Make sure they are in the same directory."

    email_features = extract_features_from_email(email_text, feature_names)
    email_features_reshaped = email_features.reshape(1, -1)
    email_df = pd.DataFrame(email_features_reshaped, columns=feature_names)
    email_features_scaled = scaler.transform(email_df)
    prediction = model.predict(email_features_scaled)


    return "Spam" if prediction[0] == 1 else "Not Spam"

if __name__ == '__main__':
    # Example usage:
    sample_email_spam = """
    Subject: Congratulations! You've Won a FREE Trip!

    Dear Winner,

    You have been selected to receive a FREE all-inclusive vacation to a tropical paradise! 
    This is a limited-time offer, so you must act now. To claim your prize, simply click the link below and enter your credit card details for verification.
    This is not a scam, we are a legitimate business. Your information is 100% secure.
    Click here: http://totally-not-a-scam-link.com/claim-prize
    Don't miss out on this incredible opportunity! Make all your friends jealous. Free money is waiting for you.

    Best regards,
    The Prize Team
    000-123-4567
    """

    sample_email_not_spam = """
    Subject: Project Update and Meeting Schedule

    Hi Team,

    Just a reminder about our project meeting scheduled for tomorrow at 10:00 AM in the main conference room. 
    Please review the attached document outlining the project's progress and come prepared to discuss the next steps.
    The latest data analysis report from the lab is also included. George will be presenting his findings.

    Let me know if you have any questions.

    Thanks,
    Alex
    """

    # Test the spam email
    result_spam = classify_email(sample_email_spam)
    print(f"The first email is classified as: {result_spam}")

    # Test the not-spam email
    result_not_spam = classify_email(sample_email_not_spam)
    print(f"The second email is classified as: {result_not_spam}")
