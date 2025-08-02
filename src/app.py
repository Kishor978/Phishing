import streamlit as st
import re
import torch
import json
import os

# Import URL model components
from urls.models.fusion_model import FusionModel
from urls.utils.urls_preprocessing import encode_url_char, url_to_graph
from urls.utils.urls_preprocessing import MAX_URL_LEN as URL_MAX_URL_LEN # Use specific max_len

# Import Email model components
from email_classification.models.dual_encoder_transformer import DualEncoderAttentionFusion
from email_classification.preprocessing import clean_text, preprocess_text_spacy
from email_classification.email_utils import setup_logging, device # Use utilities from email_classification/src/utils
from email_classification.model_config import DUAL_TRANS_EMBED_DIM, DUAL_TRANS_HIDDEN_DIM

# Setup logging (optional, for debugging in console/file, Streamlit handles basic output)
logger = setup_logging(log_file_name="streamlit_app_log.log")

# --- Configuration for Model Paths ---
URL_MODEL_DIR = r"E:\Phising_detection\results\models\urls"
EMAIL_MODEL_DIR = r"E:\Phising_detection\results\models\email"

# --- Model Loading Functions (cached by Streamlit) ---
@st.cache_resource
def load_url_charcgnn_model_and_assets():
    """Loads the trained URL CharCGNN (Fusion) model and its vocabularies."""
    model_path = os.path.join(URL_MODEL_DIR, 'best_fusion_model.pt')
    char_vocab_path = os.path.join(URL_MODEL_DIR, 'vocab_char_fusion.json')
    gnn_vocab_path = os.path.join(URL_MODEL_DIR, 'vocab_gnn_fusion.json')

    if not all(os.path.exists(p) for p in [model_path, char_vocab_path, gnn_vocab_path]):
        st.error(f"Required URL CharCGNN assets not found in {URL_MODEL_DIR}. Please train the model first.")
        return None, None, None

    with open(char_vocab_path, 'r') as f:
        char_vocab = json.load(f)
    with open(gnn_vocab_path, 'r') as f:
        gnn_vocab = json.load(f)

    model = FusionModel(cnn_vocab_size=len(char_vocab), gnn_vocab_size=len(gnn_vocab)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set to evaluation mode
    logger.info(f"URL CharCGNN model loaded from {model_path}")
    return model, char_vocab, gnn_vocab

@st.cache_resource
def load_email_dual_encoder_transformer_model_and_assets():
    """Loads the trained Email Dual Encoder Transformer model and its vocabulary."""
    model_path = os.path.join(EMAIL_MODEL_DIR, 'dual_transformer_best_model.pt')
    vocab_path = os.path.join(EMAIL_MODEL_DIR, 'dual_transformer_vocab.json')

    if not all(os.path.exists(p) for p in [model_path, vocab_path]):
        st.error(f"Required Email Dual Encoder Transformer assets not found in {EMAIL_MODEL_DIR}. Please train the model first.")
        return None, None

    with open(vocab_path, 'r') as f:
        vocab = json.load(f)

    # Instantiate the model using the parameters from your initialization
    model = DualEncoderAttentionFusion(
        vocab_size=len(vocab),
        embed_dim=DUAL_TRANS_EMBED_DIM,
        hidden_dim=DUAL_TRANS_HIDDEN_DIM,
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set to evaluation mode
    logger.info(f"Email Dual Encoder Transformer model loaded from {model_path}")
    return model, vocab

# --- Prediction Functions (from inference_main.py, slightly adapted) ---
def predict_url_charcgnn(model, char_vocab, gnn_vocab, url_string):
    """Makes a prediction using the URL CharCGNN model."""
    char_encoded = encode_url_char(url_string, max_len=URL_MAX_URL_LEN, char2idx=char_vocab)
    char_input = torch.tensor(char_encoded, dtype=torch.long).unsqueeze(0).to(device) # Add batch dimension

    graph_data = url_to_graph(url_string, label=0, vocab=gnn_vocab) # Label is dummy for inference
    if graph_data is None:
        logger.warning(f"Could not create a valid graph for URL: {url_string}. Assuming legitimate for URL model.")
        return "Legitimate", 0.5 # Fallback

    graph_data = graph_data.to(device)

    with torch.no_grad():
        logits = model(char_input, graph_data)
        probability = torch.sigmoid(logits).item() # Get probability from logits

    prediction = "Phishing" if probability > 0.5 else "Legitimate"
    return prediction, probability

def _email_encode_text(text, vocab, max_len):
    """Helper to preprocess and encode email text for the dual encoder."""
    processed_text = preprocess_text_spacy(clean_text(text))
    tokens = processed_text.split()
    encoded = [vocab.get(token, vocab['<UNK>']) for token in tokens]

    if len(encoded) < max_len:
        padded_text = encoded + [vocab['<PAD>']] * (max_len - len(encoded))
    else:
        padded_text = encoded[:max_len]
    return torch.tensor(padded_text, dtype=torch.long)

def predict_email_dual_encoder_transformer(model, vocab, subject_string, body_string):
    """Makes a prediction using the Email Dual Encoder Transformer model."""
    MAX_LEN_SUBJECT = 100 
    MAX_LEN_BODY = 1000 
    
    encoded_subject = _email_encode_text(subject_string, vocab, MAX_LEN_SUBJECT).unsqueeze(0).to(device)
    encoded_body = _email_encode_text(body_string, vocab, MAX_LEN_BODY).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(encoded_subject, encoded_body)
        probability = torch.sigmoid(logits).item()

    prediction = "Phishing" if probability > 0.5 else "Legitimate"
    return prediction, probability

# --- URL Extraction Helper ---
def extract_urls(text):
    """Extracts URLs from a given text using a regex."""
    # Regex for common URL patterns (http, https, www. followed by domain)
    url_pattern = r'https?://(?:www\.)?[a-zA-Z0-9./-]+(?:\?|\#|&)?(?:[a-zA-Z0-9./&=%_+-]+)?'
    urls = re.findall(url_pattern, text)
    return urls

# --- Streamlit UI ---
st.set_page_config(page_title="Phishing Email Detector", layout="centered")
st.title("🛡️ Phishing Email Detector")
st.markdown("Enter an email's subject and body to check if it's a phishing attempt.")

# Load models and assets (cached for performance)
url_model, url_char_vocab, url_gnn_vocab = load_url_charcgnn_model_and_assets()
email_model, email_vocab = load_email_dual_encoder_transformer_model_and_assets()

if url_model is None or email_model is None:
    st.warning("Models could not be loaded. Please ensure training was successful and model files are in the correct 'results/models' directories.")
    st.stop() # Stop the app if models aren't loaded

subject = st.text_input("Email Subject:", placeholder="e.g., Urgent Account Verification")
body = st.text_area("Email Body:", height=250, placeholder="e.g., Dear customer, your account requires immediate verification...")

if st.button("Analyze Email"):
    if not subject and not body:
        st.warning("Please enter at least the email subject or body to analyze.")
    else:
        with st.spinner("Analyzing..."):
            email_content_phishing = False
            url_phishing_detected = False
            
            email_model_pred, email_model_prob = predict_email_dual_encoder_transformer(email_model, email_vocab, subject, body)
            if email_model_pred == "Phishing":
                email_content_phishing = True
                st.info(f"Email Content Analysis: **Phishing** (Confidence: {email_model_prob:.2f})")
            else:
                st.info(f"Email Content Analysis: **Legitimate** (Confidence: {email_model_prob:.2f})")

            # Extract and analyze URLs
            extracted_urls = extract_urls(body)
            if extracted_urls:
                st.subheader("URLs Found in Email Body:")
                for i, url in enumerate(extracted_urls):
                    url_pred, url_prob = predict_url_charcgnn(url_model, url_char_vocab, url_gnn_vocab, url)
                    if url_pred == "Phishing":
                        url_phishing_detected = True
                        st.write(f"- URL {i+1}: `{url}` -> **Phishing** (Confidence: {url_prob:.2f})")
                    else:
                        st.write(f"- URL {i+1}: `{url}` -> Legitimate (Confidence: {url_prob:.2f})")
            else:
                st.info("No URLs found in the email body.")

            st.markdown("---")
            st.subheader("Overall Email Verdict:")

            if email_content_phishing and url_phishing_detected:
                st.error("🚨 This is a **Phishing Email**! Both the email content and embedded URLs indicate phishing.")
            elif email_content_phishing:
                st.error("⚠️ This email is likely **Phishing** based on its content.")
            elif url_phishing_detected:
                st.error("⚠️ This email contains **Phishing URLs**, even if the text itself seems okay.")
            else:
                st.success("✅ This email appears **Legitimate**.")

st.markdown("---")
st.markdown("This tool uses a Dual Encoder Transformer model for email content analysis and a CharCGNN model for URL analysis.")