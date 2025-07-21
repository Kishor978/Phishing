import re
import string
import spacy
from email_classification.model_config import SPACY_MODEL, SPACY_MAX_LENGTH
from email_classification.email_utils import setup_logging

logger = setup_logging()

# Load SpaCy model once
try:
    nlp = spacy.load(SPACY_MODEL, disable=["ner", "parser"])
    nlp.max_length = SPACY_MAX_LENGTH  # Increase limit
    logger.info(
        f"SpaCy model '{SPACY_MODEL}' loaded with max_length {SPACY_MAX_LENGTH}."
    )
except OSError:
    logger.error(
        f"SpaCy model '{SPACY_MODEL}' not found. Please run 'python -m spacy download {SPACY_MODEL}'"
    )
    # Fallback or exit if spacy model is essential
    nlp = None  # Set to None if loading fails


def clean_text(text):
    """
    Cleans a given text by converting to string, handling encoding,
    removing newlines/tabs, and multiple spaces.
    """
    text = str(text).encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    text = re.sub(r"[\r\n\t]+", " ", text)  # Replace newlines/tabs with space
    text = re.sub(
        r"\s+", " ", text
    ).strip()  # Replace multiple spaces with single space
    text = text.lower()  # Convert to lowercase
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )  # Remove punctuation
    return text


def preprocess_text_spacy(text):
    """
    Preprocesses text using SpaCy for tokenization, lemmatization,
    and stop word/non-alphabetic removal.
    Returns a space-separated string of processed tokens.
    """
    if nlp is None:
        logger.error("SpaCy model not loaded, skipping advanced preprocessing.")
        return text  # Return original text or empty string

    doc = nlp(text)
    # Lemmatize, remove stop words, remove non-alphabetic tokens
    tokens = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)
