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
# Import the new sentiment analysis component
from email_classification.sentiment_analysis import PhishingSentimentAnalyzer

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

    prediction = "Phishing" if probability > 0.6 else "Legitimate"
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
    # Multiple patterns for different URL formats
    
    # Standard URLs with http(s) protocol
    standard_pattern = r'https?://(?:www\.)?[a-zA-Z0-9./-]+(?:\?|\#|&)?(?:[a-zA-Z0-9./&=%_+-]+)?'
    
    # URLs without protocol (e.g., example.com/path)
    no_protocol_pattern = r'(?<!\S)(?:www\.)?[a-zA-Z0-9][a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:\/[a-zA-Z0-9\/_.-]+)?(?:\?[a-zA-Z0-9=%&_.-]+)?'
    
    # URLs enclosed in brackets or parentheses (common in phishing examples)
    bracketed_pattern = r'\[([a-zA-Z0-9][a-zA-Z0-9-]+\.[a-zA-Z]{2,}(?:\/[a-zA-Z0-9\/_.-]+)?(?:\?[a-zA-Z0-9=%&_.-]+)?)\]'
    
    # Find all matches from different patterns
    standard_urls = re.findall(standard_pattern, text)
    no_protocol_urls = re.findall(no_protocol_pattern, text)
    bracketed_urls = re.findall(bracketed_pattern, text)
    
    # Combine all results, removing duplicates
    all_urls = list(set(standard_urls + no_protocol_urls + bracketed_urls))
    
    # For URLs without protocol, add 'http://' for analysis
    for i, url in enumerate(all_urls):
        if not url.startswith(('http://', 'https://')):
            all_urls[i] = 'http://' + url
    
    return all_urls

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
            
            # Main model prediction
            email_model_pred, email_model_prob = predict_email_dual_encoder_transformer(email_model, email_vocab, subject, body)
            
            # Sentiment analysis (new component)
            sentiment_analyzer = PhishingSentimentAnalyzer()
            sentiment_results = sentiment_analyzer.analyze(subject, body)
            
            # Display main model results
            if email_model_pred == "Phishing":
                email_content_phishing = True
                st.error(f"Email Content Analysis: **Phishing** (Confidence: {email_model_prob:.2f})")
            else:
                st.success(f"Email Content Analysis: **Legitimate** (Confidence: {email_model_prob:.2f})")
                
            # Display sentiment analysis results in an expandable section
            with st.expander("Email Sentiment Analysis"):
                st.subheader("Emotional Manipulation Detection")
                
                # Create a horizontal bar chart for emotional triggers
                triggers = {
                    "Urgency": sentiment_results["urgency"],
                    "Fear": sentiment_results["fear"],
                    "Reward": sentiment_results["reward"],
                    "Trust": sentiment_results["trust"],
                    "Negative": sentiment_results["negative"]
                }
                
                # Display the scores as a progress bar
                for trigger_name, score in triggers.items():
                    st.metric(label=trigger_name, value=f"{score:.2f}")
                    st.progress(min(score, 1.0))  # Cap at 1.0 for progress bar
                
                # Show explanations
                st.subheader("Analysis")
                explanations = sentiment_analyzer.get_explanation(sentiment_results)
                for explanation in explanations:
                    st.info(explanation)

            # Extract and analyze URLs
            extracted_urls = extract_urls(body)
            if extracted_urls:
                st.subheader("URLs Found in Email Body:")
                
                # Create a container for URL analysis
                url_container = st.container()
                
                with url_container:
                    st.write("Found and analyzing the following URLs:")
                    
                    # Create columns for URL display
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write("**URL**")
                    with col2:
                        st.write("**Verdict**")
                    with col3:
                        st.write("**Confidence**")
                    
                    # Analyze each URL
                    for i, url in enumerate(extracted_urls):
                        # Display original URL (the one displayed to user might have added http://)
                        display_url = url
                        if url.startswith("http://") and "http://" not in body and "https://" not in body:
                            # If we added the protocol for analysis but it wasn't in original text
                            display_url = url[7:]  # Remove 'http://'
                        
                        # Analyze URL
                        url_pred, url_prob = predict_url_charcgnn(url_model, url_char_vocab, url_gnn_vocab, url)
                        
                        # Set flag if phishing detected
                        if url_pred == "Phishing":
                            url_phishing_detected = True
                        
                        # Display in columns with appropriate styling
                        col1, col2, col3 = st.columns([2, 1, 1])
                        with col1:
                            st.write(f"`{display_url}`")
                        with col2:
                            if url_pred == "Phishing":
                                st.markdown(f"<span style='color:red'>**{url_pred}**</span>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<span style='color:green'>**{url_pred}**</span>", unsafe_allow_html=True)
                        with col3:
                            st.write(f"{url_prob:.2f}")
                        
                        # Add horizontal divider between URLs
                        if i < len(extracted_urls) - 1:
                            st.markdown("---")
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