import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader as GeoDataLoader # For GNN/Fusion datasets
from torch.utils.data import DataLoader as TorchDataLoader   # For CharCNN dataset
import joblib # For saving/loading sklearn models and vectorizers
import json # To save/load vocab if needed
import argparse # For command-line argument parsing
import logging # For logging to terminal
from sklearn.feature_extraction.text import TfidfVectorizer # For baseline
from sklearn.ensemble import RandomForestClassifier # For baseline
from sklearn.metrics import classification_report # For baseline
import os # For creating log directory

# Import all modularized components
from urls.utils.urls_preprocessing import (
     build_char_vocab_from_df,
    build_vocab_gnn
)
from urls.utils.dataset_utils import URLCharDataset, URLGraphDataset, URLFusionDataset
from urls.models import FusionModel,URLGNN,CharCNN
from urls.utils.training_utils import train_model, device # Import the device from training_utils
from urls.utils.plotting_utils import plot_metrics, plot_confusion_matrix, plot_roc_curve # Removed plot_f1_curve as train_model history doesn't track it by default for now

# --- Configuration Constants ---
DATA_PATH = r"E:\Phising_detection\dataset\urls\urls.json"
# DATA_PATH="/kaggle/input/phishing/urls/urls.json"
MAX_VOCAB_SIZE_GNN = 5000
MAX_URL_LEN_CHARCNN = 200 # Max URL length for charCNN (defined in url_processing but good to have here too)
BATCH_SIZE = 64
EPOCHS = 30
PATIENCE = 5
LEARNING_RATE = 1e-3
LOG_DIR = 'logs'
PLOT_DIR = 'plots'
LOG_FILE = os.path.join(LOG_DIR, 'experiment_log.log')

os.makedirs(PLOT_DIR, exist_ok=True) # Ensure plot directory exists
# --- Configure Logging ---
os.makedirs(LOG_DIR, exist_ok=True) # Ensure log directory exists

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO) # Set the minimum level of messages to handle

# Create handlers
c_handler = logging.StreamHandler() # Console handler
f_handler = logging.FileHandler(LOG_FILE) # File handler

# Set levels for handlers
c_handler.setLevel(logging.INFO)
f_handler.setLevel(logging.INFO)

# Create formatters and add it to handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler.setFormatter(formatter)
f_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

# --- Helper to load data ---
def load_data(path=DATA_PATH):
    logger.info(f"Loading data from {path}...")
    df = pd.read_json(path)
    # Ensure 'text' column is string and drop NaNs/duplicates
    df['text'] = df['text'].astype(str)
    df.dropna(subset=['text', 'label'], inplace=True)
    df.drop_duplicates(subset=['text'], inplace=True)
    logger.info(f"Data loaded. Shape: {df.shape}")
    return df

# --- Function to run the GNN experiment ---
def run_gnn_experiment():
    logger.info("--- Running GNN Experiment ---")
    df = load_data()
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    logger.info(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

    vocab_gnn = build_vocab_gnn(train_df, max_vocab_size=MAX_VOCAB_SIZE_GNN)
    logger.info(f"GNN Vocabulary size: {len(vocab_gnn)}")

    try:
        train_data = URLGraphDataset(train_df, vocab_gnn)
        val_data = URLGraphDataset(val_df, vocab_gnn)
    except ValueError as e:
        logger.error(f"Error creating GNN datasets: {e}. Skipping GNN experiment.")
        return

    train_loader = GeoDataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = GeoDataLoader(val_data, batch_size=BATCH_SIZE)
    logger.info(f"GNN DataLoaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = URLGNN(vocab_size=len(vocab_gnn)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)
    criterion = nn.BCELoss() # Expects probabilities, ensure model returns sigmoid output or use BCELossWithLogits

    logger.info("Starting GNN model training...")
    model, history = train_model(
        model, train_loader, val_loader, optimizer, criterion,
        scheduler=scheduler, epochs=1, patience=PATIENCE,
        model_save_path="best_gnn_model.pt", model_type="gnn"
    )
    logger.info("GNN model training complete.")

    logger.info("\n--- GNN Evaluation ---")
    # For plotting, use the history's recorded metrics
    if history['val_labels']: # Check if validation was performed
        plot_metrics(history, title_suffix="(GNN)")
        plot_confusion_matrix(history['val_labels'], history['val_preds'], title_suffix="(GNN)")
        plot_roc_curve(history['val_labels'], history['val_probs'], title_suffix="(GNN)")
    else:
        logger.warning("No validation data or insufficient data for plotting GNN evaluation metrics.")


    # Save GNN vocabulary
    with open("vocab_gnn.json", "w") as f:
        json.dump(vocab_gnn, f)
    logger.info("GNN Model and vocabulary saved to 'best_gnn_model.pt' and 'vocab_gnn.json'.")


# --- Function to run the CharCNN experiment ---
def run_charcnn_experiment():
    logger.info("\n--- Running CharCNN Experiment ---")
    df = load_data()
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    logger.info(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

    char_vocab = build_char_vocab_from_df(train_df, max_len=MAX_URL_LEN_CHARCNN)
    logger.info(f"CharCNN Vocabulary size: {len(char_vocab)}")

    train_set = URLCharDataset(train_df)
    val_set = URLCharDataset(val_df)

    train_loader = TorchDataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = TorchDataLoader(val_set, batch_size=BATCH_SIZE)
    logger.info(f"CharCNN DataLoaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")


    model = CharCNN(vocab_size=len(char_vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=1, factor=0.5)
    criterion = nn.BCELoss() # Use BCELossWithLogits if your model returns logits

    logger.info("Starting CharCNN model training...")
    model, history = train_model(
        model, train_loader, val_loader, optimizer, criterion,
        scheduler=scheduler, epochs=EPOCHS, patience=PATIENCE,
        model_save_path="best_charcnn_model.pt", model_type="charcnn"
    )
    logger.info("CharCNN model training complete.")

    logger.info("\n--- CharCNN Evaluation ---")
    if history['val_labels']:
        plot_metrics(history, title_suffix="(CharCNN)")
        plot_confusion_matrix(history['val_labels'], history['val_preds'], title_suffix="(CharCNN)")
        plot_roc_curve(history['val_labels'], history['val_probs'], title_suffix="(CharCNN)")
    else:
        logger.warning("No validation data or insufficient data for plotting CharCNN evaluation metrics.")

    with open("vocab_char.json", "w") as f:
        json.dump(char_vocab, f)
    logger.info("CharCNN Model and vocabulary saved to 'best_charcnn_model.pt' and 'vocab_char.json'.")


# --- Function to run the Fusion experiment ---
def run_fusion_experiment():
    logger.info("\n--- Running Fusion Model Experiment ---")
    df = load_data()
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    logger.info(f"Training samples: {len(train_df)}, Validation samples: {len(val_df)}")

    char_vocab = build_char_vocab_from_df(train_df, max_len=MAX_URL_LEN_CHARCNN)
    gnn_vocab = build_vocab_gnn(train_df, max_vocab_size=MAX_VOCAB_SIZE_GNN)
    logger.info(f"CharCNN Vocab size: {len(char_vocab)}, GNN Vocab size: {len(gnn_vocab)}")

    try:
        train_data = URLFusionDataset(train_df, char_vocab, gnn_vocab, max_len=MAX_URL_LEN_CHARCNN)
        val_data = URLFusionDataset(val_df, char_vocab, gnn_vocab, max_len=MAX_URL_LEN_CHARCNN)
    except ValueError as e:
        logger.error(f"Error creating Fusion datasets: {e}. Skipping Fusion experiment.")
        return

    train_loader = GeoDataLoader(list(train_data), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = GeoDataLoader(list(val_data), batch_size=BATCH_SIZE)
    logger.info(f"Fusion DataLoaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    model = FusionModel(cnn_vocab_size=len(char_vocab), gnn_vocab_size=len(gnn_vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2)
    criterion = nn.BCELoss()

    logger.info("Starting Fusion model training...")
    model, history = train_model(
        model, train_loader, val_loader, optimizer, criterion,
        scheduler=scheduler, epochs=EPOCHS, patience=PATIENCE,
        model_save_path="best_fusion_model.pt", model_type="fusion"
    )
    logger.info("Fusion model training complete.")

    logger.info("\n--- Fusion Model Evaluation ---")
    if history['val_labels']:
        plot_metrics(history, title_suffix="(Fusion)")
        plot_confusion_matrix(history['val_labels'], history['val_preds'], title_suffix="(Fusion)")
        plot_roc_curve(history['val_labels'], history['val_probs'], title_suffix="(Fusion)")
    else:
        logger.warning("No validation data or insufficient data for plotting Fusion evaluation metrics.")

    with open("vocab_char_fusion.json", "w") as f:
        json.dump(char_vocab, f)
    with open("vocab_gnn_fusion.json", "w") as f:
        json.dump(gnn_vocab, f)
    logger.info("Fusion Model and vocabularies saved to 'best_fusion_model.pt', 'vocab_char_fusion.json', 'vocab_gnn_fusion.json'.")


# --- Function to run the Baseline (TF-IDF + RandomForest) experiment ---
def run_baseline_experiment():
    logger.info("\n--- Running Baseline (TF-IDF + RandomForest) Experiment ---")
    df = load_data()
    df['text'] = df['text'].astype(str)
    X = df['text']
    y = df['label']

    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 6), max_features=3000)
    X_tfidf = vectorizer.fit_transform(X)
    logger.info(f"TF-IDF features extracted. Shape: {X_tfidf.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, stratify=y, random_state=42
    )
    logger.info(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    logger.info("Starting Baseline model training...")
    clf.fit(X_train, y_train)
    logger.info("Baseline model training complete.")

    logger.info("\n--- Baseline Evaluation ---")
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1] # Probability of the positive class

    logger.info("Classification Report (Baseline):\n" + classification_report(y_test, y_pred))

    plot_confusion_matrix(y_test, y_pred, title_suffix="(Baseline)")
    plot_roc_curve(y_test, y_prob, title_suffix="(Baseline)")

    joblib.dump(clf, "baseline_rf_model.pkl")
    joblib.dump(vectorizer, "baseline_tfidf_vectorizer.pkl")
    logger.info("Baseline Model and vectorizer saved to 'baseline_rf_model.pkl' and 'baseline_tfidf_vectorizer.pkl'.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run URL Phishing Detection Experiments.")
    parser.add_argument(
        '--model',
        type=str,
        choices=['all', 'gnn', 'charcnn', 'fusion', 'baseline'],
        default='all',
        help="Specify which model experiment to run. 'all' runs all experiments."
    )
    args = parser.parse_args()

    logger.info(f"Starting experiments for model: {args.model}")

    if args.model == 'all':
        run_gnn_experiment()
        run_charcnn_experiment()
        run_fusion_experiment()
        run_baseline_experiment()
    elif args.model == 'gnn':
        run_gnn_experiment()
    elif args.model == 'charcnn':
        run_charcnn_experiment()
    elif args.model == 'fusion':
        run_fusion_experiment()
    elif args.model == 'baseline':
        run_baseline_experiment()
    else:
        logger.error(f"Invalid model choice: {args.model}")

    logger.info("\nAll selected experiments completed. Check 'logs/experiment_log.log' for detailed output.")