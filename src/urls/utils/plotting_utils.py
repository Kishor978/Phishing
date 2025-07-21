import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import os

PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)  # Ensure plot directory exists


def plot_metrics(history, title_suffix=""):
    """
    Plots training and validation loss and accuracy over epochs.
    """
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title(f"Loss Over Epochs {title_suffix}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.title(f"Accuracy Over Epochs {title_suffix}")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{title_suffix} plot.png"))

    plt.show()


def plot_f1_curve(history, title_suffix=""):
    """
    Plots training and validation F1 score over epochs.
    (Applicable if you track train_f1 in history).
    """
    if "f1" in history and "val_f1" in history:
        plt.figure(figsize=(6, 5))
        plt.plot(history["f1"], label="Train F1")
        plt.plot(history["val_f1"], label="Val F1")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.title(f"Training vs Validation F1 Score {title_suffix}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{title_suffix} f1_plot.png"))

        plt.show()
    else:
        print("F1 scores not available in history for plotting.")


def plot_confusion_matrix(y_true, y_pred, title_suffix=""):
    """
    Plots a confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legit", "Phish"],
        yticklabels=["Legit", "Phish"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix {title_suffix}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{title_suffix} confusion_matrix.png"))

    plt.show()


def plot_roc_curve(y_true, y_probs, title_suffix=""):
    """
    Plots the ROC curve and displays AUC.
    """
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve {title_suffix}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, f"{title_suffix} roc_curve.png"))
    plt.show()
