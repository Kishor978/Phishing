import pandas as pd
import csv
import sys
import os
from email_classification.model_config import (
    DATA_PATH_TREC_05,
    DATA_PATH_TREC_06,
    DATA_PATH_TREC_07,
    PROCESSED_DATA_PATH,
)
from email_classification.email_utils import setup_logging
import re

logger = setup_logging()

csv.field_size_limit(1_000_000_000)


def load_and_merge_trec_data():
    """
    Loads and merges TREC email datasets from multiple CSV files.
    Performs initial cleaning by dropping NaNs and duplicates.
    Saves the merged DataFrame to a processed data path.
    """
    if os.path.exists(PROCESSED_DATA_PATH):
        logger.info(f"Loading merged data from {PROCESSED_DATA_PATH}...")
        df = pd.read_csv(PROCESSED_DATA_PATH)
        logger.info(f"Merged data loaded. Shape: {df.shape}")
        return df

    logger.info("Loading and merging TREC email datasets...")
    dataframes = []
    file_paths = [DATA_PATH_TREC_05, DATA_PATH_TREC_06, DATA_PATH_TREC_07]

    for f_path in file_paths:
        if not os.path.exists(f_path):
            logger.warning(f"File not found: {f_path}. Skipping.")
            continue
        try:
            # Assuming 'text' for body and 'label' for classification
            # Some notebooks infer 'subject' and 'body'
            # Let's align with the BiLSTM notebooks that use 'text' as combined for features,
            # but RandomForest needs separate subject/body.
            # The TREC dataset actually has 'subject' and 'body' columns.
            df = pd.read_csv(
                f_path,
                usecols=["text", "label"],
                encoding="latin-1",
                engine="python",
                on_bad_lines="skip",
            )
            df.columns = ["text", "label"]  # Ensure consistent column names
            dataframes.append(df)
            logger.info(f"Loaded {f_path}. Shape: {df.shape}")
        except Exception as e:
            logger.error(f"Error loading {f_path}: {e}")

    if not dataframes:
        logger.error("No dataframes were loaded. Check file paths and formats.")
        return pd.DataFrame()

    merged_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"Initial merged data shape: {merged_df.shape}")

    # Drop rows where 'text' or 'label' are NaN
    initial_rows = merged_df.shape[0]
    merged_df.dropna(subset=["text", "label"], inplace=True)
    logger.info(
        f"Dropped {initial_rows - merged_df.shape[0]} rows with NaN values. New shape: {merged_df.shape}"
    )

    # Drop duplicates based on 'text'
    initial_rows = merged_df.shape[0]
    merged_df.drop_duplicates(subset=["text"], inplace=True)
    logger.info(
        f"Dropped {initial_rows - merged_df.shape[0]} duplicate rows. Final shape: {merged_df.shape}"
    )

    # Convert label to numeric (assuming 'spam'/'ham' or similar)
    # The notebooks imply 0/1 labels. Let's map 'spam' to 1, 'ham' to 0 if they exist.
    # Check unique labels to confirm mapping
    if merged_df["label"].dtype == "object":
        unique_labels = merged_df["label"].unique()
        if "spam" in unique_labels or "ham" in unique_labels:
            merged_df["label"] = merged_df["label"].map({"ham": 0, "spam": 1})
            logger.info(f"Mapped 'ham' to 0 and 'spam' to 1 in 'label' column.")
        else:
            logger.warning(
                f"Unrecognized object labels: {unique_labels}. Assuming labels are already 0/1 or will be handled downstream."
            )
            # Attempt to convert to numeric, coerce errors to NaN and drop
            merged_df["label"] = pd.to_numeric(merged_df["label"], errors="coerce")
            merged_df.dropna(subset=["label"], inplace=True)
            merged_df["label"] = merged_df["label"].astype(int)
            logger.info(
                f"Attempted to convert labels to numeric. New shape: {merged_df.shape}"
            )

    # Save processed data for faster loading next time
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    merged_df.to_csv(PROCESSED_DATA_PATH, index=False)
    logger.info(f"Merged and processed data saved to {PROCESSED_DATA_PATH}")

    return merged_df


def extract_subject_body(df):
    """
    Extracts subject and body from the 'text' column,
    assuming 'Subject: ...\n\nBody: ...' format.
    If not found, assigns empty string or full text to body.
    """
    df_copy = df.copy()
    subjects = []
    bodies = []
    for text in df_copy["text"]:
        if isinstance(text, str):
            subject_match = re.match(r"Subject: (.*?)\n\n(.*)", text, re.DOTALL)
            if subject_match:
                subjects.append(subject_match.group(1).strip())
                bodies.append(subject_match.group(2).strip())
            else:
                subjects.append("")  # No explicit subject line found
                bodies.append(text.strip())  # Treat whole text as body
        else:
            subjects.append("")
            bodies.append("")

    df_copy["subject"] = subjects
    df_copy["body"] = bodies
    logger.info("Extracted 'subject' and 'body' columns from 'text'.")
    return df_copy
