# email_classification/src/evaluation.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import numpy as np
import os
from email_classification.email_utils import save_plot, setup_logging

logger = setup_logging()

def plot_metrics(history, title_suffix=""):
    """Plots training and validation loss, accuracy, and F1-score over epochs."""
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.title(f'Loss Over Epochs {title_suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy')
    plt.plot(epochs, history['val_acc'], label='Val Accuracy')
    plt.title(f'Accuracy Over Epochs {title_suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['train_f1'], label='Train F1-Score')
    plt.plot(epochs, history['val_f1'], label='Val F1-Score')
    plt.title(f'F1-Score Over Epochs {title_suffix}')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.legend()

    plt.tight_layout()
    save_plot(plt, f"metrics_over_epochs{title_suffix.replace('(','').replace(')','').replace(' ','_')}.png", sub_dir="metrics_curves")
    logger.info(f"Metrics plot saved for {title_suffix}")

def plot_confusion_matrix(y_true, y_pred, labels=['Legitimate', 'Phishing'], title_suffix=""):
    """Plots the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix {title_suffix}')
    plt.tight_layout()
    save_plot(plt, f"confusion_matrix{title_suffix.replace('(','').replace(')','').replace(' ','_')}.png", sub_dir="confusion_matrices")
    logger.info(f"Confusion matrix plot saved for {title_suffix}")

def plot_roc_curve(y_true, y_probs, title_suffix=""):
    """Plots the Receiver Operating Characteristic (ROC) curve."""
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_probs, np.ndarray):
        y_probs = np.array(y_probs)

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve {title_suffix}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    save_plot(plt, f"roc_curve{title_suffix.replace('(','').replace(')','').replace(' ','_')}.png", sub_dir="roc_curves")
    logger.info(f"ROC curve plot saved for {title_suffix}")

def print_and_save_classification_report(y_true, y_pred, model_name, labels=None):
    """Prints and saves the classification report."""
    report = classification_report(y_true, y_pred, target_names=labels, zero_division=0)
    logger.info(f"\nClassification Report for {model_name}:\n{report}")
    
    # Save to a text file
    report_file_path = os.path.join(os.path.join(os.path.dirname(__file__), '..', 'results', 'metrics'), f"{model_name}_classification_report.txt")
    with open(report_file_path, 'w') as f:
        f.write(f"Classification Report for {model_name}:\n")
        f.write(report)
    logger.info(f"Classification report saved to: {report_file_path}")