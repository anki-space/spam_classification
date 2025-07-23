import streamlit as st
import joblib
import numpy as np
import re
import pandas as pd


st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="auto",
)
# --- Feature Extraction Function ---
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

# --- Load Model and Scaler ---
@st.cache_resource
def load_artifacts():
    """
    Loads the saved model, scaler, and feature names from disk.
    This function is cached so the files are only loaded once per session.
    """
    try:
        model = joblib.load('models/spam_classifier_rf.joblib')
        scaler = joblib.load('models/scaler.joblib')
        feature_names = joblib.load('models/feature_names.joblib')
        return model, scaler, feature_names
    except FileNotFoundError:
        st.error("Error: Model, scaler, or feature name files not found. Make sure they are in the same directory as the app.")
        return None, None, None

model, scaler, feature_names = load_artifacts()

# --- Classification Function ---
def classify_email_streamlit(email_text, model, scaler, feature_names):
    """
    Classifies the email using the pre-loaded model and scaler.
    """
    if model is None or scaler is None or feature_names is None:
        return "Classification service is unavailable."

    email_features = extract_features_from_email(email_text, feature_names)
    email_features_reshaped = email_features.reshape(1, -1)
    email_df = pd.DataFrame(email_features_reshaped, columns=feature_names)
    email_features_scaled = scaler.transform(email_df)
    prediction = model.predict(email_features_scaled)
    return "Spam" if prediction[0] == 1 else "Not Spam"

# --- App Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #0E1117; /* Dark background */
        color: #FAFAFA; /* Light text */
    }
    .stButton>button {
        background-color: #1E88E5; /* Blue button */
        color: white;
        border-radius: 12px;
        padding: 10px 24px;
        border: none;
        font-size: 16px;
        font-weight: bold;
    }
    .stTextArea>div>div>textarea {
        background-color: #262730; /* Darker text area */
        color: #FAFAFA; /* Light text in text area */
        border-radius: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- App Header ---
st.title("📧 Spam Email Classifier")
st.write(
    "Paste the full content of an email below (including headers like 'Subject:') "
    "to check if it's spam or not."
)

# --- Email Input Area ---
email_text = st.text_area(
    "Email Content",
    height=300,
    placeholder="Subject: Special Offer!..."
)

# --- Classification Button and Logic ---
if st.button("Classify Email"):
    if email_text:
        with st.spinner('Analyzing the email...'):
            result = classify_email_streamlit(email_text, model, scaler, feature_names)

        # Display the result
        if "Spam" in result:
            st.error(f"**Classification Result: {result}** 📮", icon="🚨")
        elif "unavailable" in result:
            st.error(result, icon="❌")
        else:
            st.success(f"**Classification Result: {result}** ✅", icon="👍")
    else:
        st.warning("Please paste some email text to classify.", icon="⚠️")

