import argparse
import pandas as pd
import torch.optim as optim
import json
import os
from sklearn.model_selection import train_test_split
import torch
from sklearn.metrics import roc_auc_score

# Import modules from email
from email_classification.email_utils import (
    setup_logging,
    device,
    MODELS_DIR,
    save_metrics,
)
from email_classification.preprocessing import clean_text, preprocess_text_spacy
from email_classification.datasets import EmailDataset, DualEncoderEmailDataset
from email_classification.models import (
    TraditionalEmailClassifier,
    BiLSTMClassifier,
    DualEncoderBilstmFusion,
    DualEncoderAttentionFusion,
)
from email_classification.training import train_pytorch_model, train_traditional_model
from email_classification.evaluation import (
    plot_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
    print_and_save_classification_report,
)
from email_classification.model_config import (
    EPOCHS,
    PATIENCE,
    LEARNING_RATE,
    BATCH_SIZE,
    BILSTM_EMBED_DIM,
    BILSTM_HIDDEN_DIM,
    BILSTM_NUM_LAYERS,
    BILSTM_DROPOUT,
    BILSTM_BIDIRECTIONAL,
    DUAL_BILSTM_EMBED_DIM,
    DUAL_BILSTM_HIDDEN_DIM,
    DUAL_BILSTM_NUM_LAYERS,
    DUAL_BILSTM_DROPOUT,
    DUAL_BILSTM_BIDIRECTIONAL,
    DUAL_BILSTM_FUSION_HIDDEN_DIM,
    DUAL_BILSTM_FUSION_DROPOUT,
    DUAL_TRANS_EMBED_DIM,
    DUAL_TRANS_HIDDEN_DIM,
)

# Setup logging
logger = setup_logging()


def run_experiment(model_type):
    logger.info(f"--- Starting {model_type.replace('_', ' ').title()} Experiment ---")

    # 1. Load Data
    # df = load_and_merge_trec_data()
    # df = pd.read_csv(r'E:\Phising_detection\dataset\emails\TREC_07.csv',engine='python')
    df = pd.read_csv("/kaggle/input/phishing/emails/TREC_07.csv", engine="python")
    df.dropna(subset=["subject", "body", "label"], inplace=True)

    if df.empty:
        logger.error("No data loaded. Exiting experiment.")
        return
    # 2. Preprocess Data (apply to all relevant columns)
    logger.info("Applying preprocessing to email texts...")

    df["cleaned_subject"] = df["subject"].apply(clean_text)
    df["processed_subject"] = df["cleaned_subject"].apply(preprocess_text_spacy)

    df["cleaned_body"] = df["body"].apply(clean_text)
    df["processed_body"] = df["cleaned_body"].apply(preprocess_text_spacy)
    df["processed_text"] = df["processed_subject"] + " " + df["processed_body"]
    # 3. Split Data
    # For PyTorch models, split before Dataset creation
    train_df, val_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    logger.info(f"Train samples: {len(train_df)}, Validation samples: {len(val_df)}")

    if model_type == "baseline":
        # Baseline uses raw (but cleaned) subject/body for TF-IDF
        X_train_subject = train_df["processed_subject"].tolist()
        X_train_body = train_df["processed_body"].tolist()
        y_train = train_df["label"].values

        X_val_subject = val_df["processed_subject"].tolist()
        X_val_body = val_df["processed_body"].tolist()
        y_val = val_df["label"].values

        model_wrapper = TraditionalEmailClassifier()
        history = train_traditional_model(
            model_wrapper,
            X_train_subject,
            X_train_body,
            y_train,
            X_val_subject,
            X_val_body,
            y_val,
            model_name="baseline_rf",
        )

    else:  # PyTorch models
        vocab = None
        if model_type == "single_bilstm":
            # Max length for single BiLSTM for general email text
            MAX_LEN_SINGLE_BILSTM = (
                1000  # A common value, adjust based on analysis if needed
            )
            train_dataset = EmailDataset(
                train_df, text_col="processed_text", max_len=MAX_LEN_SINGLE_BILSTM
            )
            val_dataset = EmailDataset(
                val_df,
                text_col="processed_text",
                max_len=MAX_LEN_SINGLE_BILSTM,
                vocab=train_dataset.vocab,
            )
            vocab_size = len(train_dataset.vocab)
            model = BiLSTMClassifier(
                vocab_size,
                embed_dim=BILSTM_EMBED_DIM,
                hidden_dim=BILSTM_HIDDEN_DIM,
                num_layers=BILSTM_NUM_LAYERS,
                dropout=BILSTM_DROPOUT,
                bidirectional=BILSTM_BIDIRECTIONAL,
            ).to(device)

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=BATCH_SIZE, shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=BATCH_SIZE, shuffle=False
            )

        elif model_type in ["dual_bilstm", "dual_transformer"]:
            # Max lengths for subject and body for dual encoders
            MAX_LEN_SUBJECT = 100  # Adjust based on data analysis
            MAX_LEN_BODY = 1000  # Adjust based on data analysis

            train_dataset = DualEncoderEmailDataset(
                train_df,
                subject_col="processed_subject",
                body_col="processed_body",
                max_len_subject=MAX_LEN_SUBJECT,
                max_len_body=MAX_LEN_BODY,
            )
            val_dataset = DualEncoderEmailDataset(
                val_df,
                subject_col="processed_subject",
                body_col="processed_body",
                max_len_subject=MAX_LEN_SUBJECT,
                max_len_body=MAX_LEN_BODY,
                vocab=train_dataset.vocab,
            )
            vocab_size = len(train_dataset.vocab)

            if model_type == "dual_bilstm":
                model = DualEncoderBilstmFusion(
                    vocab_size,
                    embed_dim=DUAL_BILSTM_EMBED_DIM,
                    hidden_dim=DUAL_BILSTM_HIDDEN_DIM,
                    num_layers=DUAL_BILSTM_NUM_LAYERS,
                    dropout=DUAL_BILSTM_DROPOUT,
                    bidirectional=DUAL_BILSTM_BIDIRECTIONAL,
                    fusion_hidden_dim=DUAL_BILSTM_FUSION_HIDDEN_DIM,
                    fusion_dropout=DUAL_BILSTM_FUSION_DROPOUT,
                ).to(device)
            else:  # dual_transformer
                model = DualEncoderAttentionFusion(
                    vocab_size,
                    embed_dim=DUAL_TRANS_EMBED_DIM,
                    hidden_dim=DUAL_TRANS_HIDDEN_DIM,
                ).to(device)

            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=BATCH_SIZE, shuffle=True
            )
            val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=BATCH_SIZE, shuffle=False
            )

        logger.info(f"Model: {model.__class__.__name__} initialized.")
        logger.info(
            f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}"
        )

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=PATIENCE
        )

        model, history = train_pytorch_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            model_type,
            scheduler=scheduler,
            epochs=EPOCHS,
            patience=PATIENCE,
            learning_rate=LEARNING_RATE,
        )

        # Save vocabulary for PyTorch models
        if vocab is None:  # Vocab was built by the train_dataset
            vocab_to_save = train_dataset.vocab
        else:  # Vocab was passed in (e.g. from a loaded train_dataset)
            vocab_to_save = vocab

        vocab_file_path = os.path.join(MODELS_DIR, f"{model_type}_vocab.json")
        with open(vocab_file_path, "w") as f:
            json.dump(vocab_to_save, f)
        logger.info(f"Vocabulary saved to {vocab_file_path}")

        logger.info(f"\n--- {model_type.replace('_', ' ').title()} Evaluation ---")
        if history["val_labels"]:
            plot_metrics(
                history, title_suffix=f"({model_type.replace('_',' ').title()})"
            )
            plot_confusion_matrix(
                history["val_labels"],
                history["val_preds"],
                title_suffix=f"({model_type.replace('_',' ').title()})",
            )
            plot_roc_curve(
                history["val_labels"],
                history["val_probs"],
                title_suffix=f"({model_type.replace('_',' ').title()})",
            )

            # Print and save classification report for PyTorch models
            print_and_save_classification_report(
                history["val_labels"],
                history["val_preds"],
                model_name=model_type,
                labels=["Ham", "Spam"],
            )

            # Save final metrics summary for PyTorch models
            final_metrics = {
                "final_val_loss": history["val_loss"][-1],
                "final_val_accuracy": history["val_acc"][-1],
                "final_val_f1_score": history["val_f1"][-1],
                "final_val_roc_auc": roc_auc_score(
                    history["val_labels"], history["val_probs"]
                ),
            }
            save_metrics(final_metrics, f"{model_type}_final_metrics.json")
        else:
            logger.warning(
                f"No validation data or insufficient data for plotting {model_type} evaluation metrics."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Email Phishing Detection Experiments."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["all", "baseline", "single_bilstm", "dual_bilstm", "dual_transformer"],
        default="all",
        help="Specify which email model experiment to run. 'all' runs all experiments.",
    )
    args = parser.parse_args()

    if args.model == "all":
        run_experiment("baseline")
        run_experiment("single_bilstm")
        run_experiment("dual_bilstm")
        run_experiment("dual_transformer")
    else:
        run_experiment(args.model)

    logger.info("\nAll selected email experiments completed.")
    logger.info(
        f"Detailed logs saved to {os.path.join(os.getcwd(), 'logs', 'experiment_log.log')}"
    )
    logger.info(
        f"Results (models, plots, metrics) saved to {os.path.join(os.getcwd(), 'results')}"
    )
