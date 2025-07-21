import logging
import os
import torch
import joblib
import matplotlib.pyplot as plt
import json

# Define base directories
LOG_DIR = 'logs'
RESULTS_DIR = 'results'
MODELS_DIR = os.path.join(RESULTS_DIR, 'models')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
METRICS_DIR = os.path.join(RESULTS_DIR, 'metrics')

# Ensure directories exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

def setup_logging(log_file_name="experiment_log.log"):
    """
    Sets up logging to both console and a file.
    """
    log_file_path = os.path.join(LOG_DIR, log_file_name)

    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Clear existing handlers to prevent duplicate output if called multiple times
    if (logger.hasHandlers()):
        logger.handlers.clear()

    # Create handlers
    c_handler = logging.StreamHandler() # Console handler
    f_handler = logging.FileHandler(log_file_path) # File handler

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
    return logger

def save_pytorch_model(model, path, model_name="best_model.pt"):
    """Saves a PyTorch model's state_dict."""
    full_path = os.path.join(MODELS_DIR, model_name)
    torch.save(model.state_dict(), full_path)
    logging.getLogger(__name__).info(f"PyTorch model saved to: {full_path}")

def load_pytorch_model(model_class, path, model_name="best_model.pt", **kwargs):
    """Loads a PyTorch model's state_dict."""
    full_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(full_path):
        logging.getLogger(__name__).error(f"Model file not found: {full_path}")
        return None
    model = model_class(**kwargs)
    model.load_state_dict(torch.load(full_path))
    logging.getLogger(__name__).info(f"PyTorch model loaded from: {full_path}")
    return model

def save_sklearn_model(model, path, model_name="sklearn_model.pkl"):
    """Saves a scikit-learn model."""
    full_path = os.path.join(MODELS_DIR, model_name)
    joblib.dump(model, full_path)
    logging.getLogger(__name__).info(f"Scikit-learn model saved to: {full_path}")

def load_sklearn_model(path, model_name="sklearn_model.pkl"):
    """Loads a scikit-learn model."""
    full_path = os.path.join(MODELS_DIR, model_name)
    if not os.path.exists(full_path):
        logging.getLogger(__name__).error(f"Model file not found: {full_path}")
        return None
    model = joblib.load(full_path)
    logging.getLogger(__name__).info(f"Scikit-learn model loaded from: {full_path}")
    return model

def save_plot(plt_obj, plot_name, sub_dir=""):
    """Saves a matplotlib plot."""
    plot_path = os.path.join(PLOTS_DIR, sub_dir)
    os.makedirs(plot_path, exist_ok=True)
    full_path = os.path.join(plot_path, plot_name)
    plt_obj.savefig(full_path, bbox_inches='tight')
    logging.getLogger(__name__).info(f"Plot saved to: {full_path}")
    plt_obj.close() # Close the plot to free memory

def save_metrics(metrics_dict, file_name):
    """Saves evaluation metrics to a JSON file."""
    full_path = os.path.join(METRICS_DIR, file_name)
    with open(full_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    logging.getLogger(__name__).info(f"Metrics saved to: {full_path}")

# Common device for PyTorch models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")