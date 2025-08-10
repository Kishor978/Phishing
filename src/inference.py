import argparse
import torch
import json
import os
import sys

# Import URL model components
from urls.models import CharCNN
from urls.utils.urls_preprocessing import encode_url_char, MAX_URL_LEN
from urls.utils.enhanced_classifier import EnhancedURLClassifier

# Import Email model components
from email_classification.models.dual_encoder_transformer import DualEncoderAttentionFusion
from email_classification.preprocessing import clean_text, preprocess_text_spacy
from email_classification.email_utils import setup_logging, device # Use utilities from email_classification/src/utils
from email_classification.model_config import DUAL_TRANS_EMBED_DIM, DUAL_TRANS_HIDDEN_DIM

# Setup logging for inference
logger = setup_logging(log_file_name="inference_log.log")

# --- Configuration for Model Paths ---
URL_MODEL_DIR = r"E:\Phising_detection\results\models\urls"
EMAIL_MODEL_DIR = r"E:\Phising_detection\results\models\email"

# --- URL CharCNN Inference ---
def load_url_charcnn_model_and_assets():
    """Loads the trained URL CharCNN model and its vocabulary."""
    model_path = os.path.join(URL_MODEL_DIR, 'best_charcnn.pt')
    vocab_path = os.path.join(URL_MODEL_DIR, 'vocab_char.json')

    # Try notebook directory as fallback if files don't exist in models dir
    if not all(os.path.exists(p) for p in [model_path, vocab_path]):
        nb_model_path = r"E:\Phising_detection\notebooks\url\best_charcnn.pt"
        nb_vocab_path = r"E:\Phising_detection\notebooks\url\vocab_char.json"
        
        if all(os.path.exists(p) for p in [nb_model_path, nb_vocab_path]):
            model_path = nb_model_path
            vocab_path = nb_vocab_path
        else:
            logger.error(f"Required URL CharCNN assets not found. Please train the model first.")
            return None, None

    with open(vocab_path, 'r') as f:
        char_vocab = json.load(f)

    # Instantiate the model
    model = CharCNN(vocab_size=len(char_vocab)+1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set to evaluation mode

    logger.info(f"URL CharCNN model loaded from {model_path}")
    return model, char_vocab

def predict_url_charcnn(model, char_vocab, url_string):
    """Makes a prediction using the URL CharCNN model with enhanced classification."""
    logger.info(f"Predicting for URL: {url_string}")
    # Create a simple wrapper function to use the model directly
    def get_model_prediction(url):
        # Encode the URL
        char_encoded = encode_url_char(url, max_len=MAX_URL_LEN, char2idx=char_vocab)
        char_input = torch.tensor(char_encoded, dtype=torch.long).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(char_input)
            probability = output.item()
        
        # Return prediction
        prediction = "Phishing" if probability > 0.5 else "Legitimate"
        return prediction, probability
    
    # First check if URL is in whitelist
    classifier = EnhancedURLClassifier()
    if classifier.is_known_legitimate(url_string):
        return "Legitimate", 0.95
    
    # Get model prediction
    prediction, probability = get_model_prediction(url_string)
    
    # Get URL features for rule-based enhancements
    features = classifier.extract_url_features(url_string)
    
    # Apply simple rule-based adjustments
    if prediction == "Legitimate" and (
        features.get('has_suspicious_keywords', False) or 
        features.get('domain_entropy', 0) > 4.0 or
        features.get('domain_dash_count', 0) > 2
    ):
        # Increase suspicion for legitimate URLs with suspicious features
        probability = max(0.4, probability)
    
    logger.info(f"URL Prediction: {prediction}, Probability: {probability:.4f}")
    return prediction, probability

# --- Email Dual Encoder Transformer Inference ---
def load_email_dual_encoder_transformer_model_and_assets():
    """Loads the trained Email Dual Encoder Transformer model and its vocabulary."""
    model_path = os.path.join(EMAIL_MODEL_DIR, 'dual_transformer_best_model.pt')
    vocab_path = os.path.join(EMAIL_MODEL_DIR, 'dual_transformer_vocab.json')

    if not all(os.path.exists(p) for p in [model_path, vocab_path]):
        logger.error(f"Required Email Dual Encoder Transformer assets not found in {EMAIL_MODEL_DIR}. Please train the model first.")
        return None, None

    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    model = DualEncoderAttentionFusion(
                    vocab_size=len(vocab),
                    embed_dim=DUAL_TRANS_EMBED_DIM,
                    hidden_dim=DUAL_TRANS_HIDDEN_DIM,
                ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set to evaluation mode

    logger.info(f"Email Dual Encoder Transformer model loaded from {model_path}")
    return model, vocab

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
    logger.info(f"Predicting for Email - Subject: '{subject_string[:50]}...', Body: '{body_string[:50]}...'")

    MAX_LEN_SUBJECT = 100 
    MAX_LEN_BODY = 1000   

    # Preprocess and encode
    encoded_subject = _email_encode_text(subject_string, vocab, MAX_LEN_SUBJECT).unsqueeze(0).to(device)
    encoded_body = _email_encode_text(body_string, vocab, MAX_LEN_BODY).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(encoded_subject, encoded_body)
        probability = torch.sigmoid(logits).item()

    prediction = "Phishing" if probability > 0.5 else "Legitimate"
    logger.info(f"Email Prediction: {prediction}, Probability: {probability:.4f}")
    return prediction, probability

# --- Main execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference for phishing detection models.")
    parser.add_argument('--model_type', type=str, choices=['url_charcnn', 'email_transformer'], required=True,
                        help="Specify which model to use for inference.")
    parser.add_argument('--url', type=str, help="URL string for URL model inference.")
    parser.add_argument('--email_subject', type=str, help="Email subject string for email model inference.")
    parser.add_argument('--email_body', type=str, help="Email body string for email model inference.")
    
    args = parser.parse_args()

    if args.model_type == 'url_charcnn':
        if not args.url:
            parser.error("--url is required for url_charcnn model_type.")
        
        url_model, url_char_vocab = load_url_charcnn_model_and_assets()
        if url_model:
            url_pred, url_prob = predict_url_charcnn(url_model, url_char_vocab, args.url)
            logger.info(f"\n--- URL CharCNN Inference Result ---")
            logger.info(f"Input URL: {args.url}")
            logger.info(f"Predicted Class: {url_pred}")
            logger.info(f"Confidence (Phishing Probability): {url_prob:.4f}")

    elif args.model_type == 'email_transformer':
        if not args.email_subject or not args.email_body:
            parser.error("--email_subject and --email_body are required for email_transformer model_type.")
        
        email_model, email_vocab = load_email_dual_encoder_transformer_model_and_assets()
        if email_model:
            email_pred, email_prob = predict_email_dual_encoder_transformer(email_model, email_vocab, args.email_subject, args.email_body)
            logger.info(f"\n--- Email Dual Encoder Transformer Inference Result ---")
            logger.info(f"Input Subject: {args.email_subject}")
            logger.info(f"Input Body: {args.email_body}")
            logger.info(f"Predicted Class: {email_pred}")
            logger.info(f"Confidence (Phishing Probability): {email_prob:.4f}")

    logger.info("\nInference process completed.")