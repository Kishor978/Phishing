import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack
import os

from email_classification.model_config import RF_N_ESTIMATORS, RF_RANDOM_STATE, TFIDF_MAX_FEATURES
from email_classification.email_utils import setup_logging

logger = setup_logging()

class TraditionalEmailClassifier:
    """
    A wrapper for the RandomForestClassifier with TF-IDF vectorization
    for subject and body of emails.
    """
    def __init__(self):
        self.subject_vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
        self.body_vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
        self.model = RandomForestClassifier(n_estimators=RF_N_ESTIMATORS, random_state=RF_RANDOM_STATE)
        logger.info(f"TraditionalEmailClassifier initialized with RandomForest(n_estimators={RF_N_ESTIMATORS})")
        logger.info(f"TF-IDF Vectorizers initialized with max_features={TFIDF_MAX_FEATURES}")

    def fit(self, X_train_subject, X_train_body, y_train):
        """
        Fits the TF-IDF vectorizers and the RandomForestClassifier.
        X_train_subject and X_train_body should be lists of preprocessed strings.
        """
        logger.info("Fitting TF-IDF vectorizers...")
        X_train_subject_tfidf = self.subject_vectorizer.fit_transform(X_train_subject)
        X_train_body_tfidf = self.body_vectorizer.fit_transform(X_train_body)
        
        X_train_combined = hstack([X_train_subject_tfidf, X_train_body_tfidf])
        logger.info(f"Combined TF-IDF features shape: {X_train_combined.shape}")

        logger.info("Fitting RandomForestClassifier...")
        self.model.fit(X_train_combined, y_train)
        logger.info("RandomForestClassifier training complete.")

    def predict(self, X_test_subject, X_test_body):
        """
        Transforms test data using fitted vectorizers and makes predictions.
        X_test_subject and X_test_body should be lists of preprocessed strings.
        """
        X_test_subject_tfidf = self.subject_vectorizer.transform(X_test_subject)
        X_test_body_tfidf = self.body_vectorizer.transform(X_test_body)
        
        X_test_combined = hstack([X_test_subject_tfidf, X_test_body_tfidf])
        return self.model.predict(X_test_combined)

    def predict_proba(self, X_test_subject, X_test_body):
        """
        Transforms test data using fitted vectorizers and gets probability estimates.
        """
        X_test_subject_tfidf = self.subject_vectorizer.transform(X_test_subject)
        X_test_body_tfidf = self.body_vectorizer.transform(X_test_body)
        
        X_test_combined = hstack([X_test_subject_tfidf, X_test_body_tfidf])
        return self.model.predict_proba(X_test_combined)[:, 1] # Probability of the positive class

    def save_assets(self, base_path):
        """Saves the model and vectorizers."""
        joblib.dump(self.model, os.path.join(base_path, "rf_email_model.pkl"))
        joblib.dump(self.subject_vectorizer, os.path.join(base_path, "tfidf_subject_vectorizer.pkl"))
        joblib.dump(self.body_vectorizer, os.path.join(base_path, "tfidf_body_vectorizer.pkl"))
        logger.info(f"RandomForest model and TF-IDF vectorizers saved to {base_path}")

    @classmethod
    def load_assets(cls, base_path):
        """Loads the model and vectorizers."""
        instance = cls()
        instance.model = joblib.load(os.path.join(base_path, "rf_email_model.pkl"))
        instance.subject_vectorizer = joblib.load(os.path.join(base_path, "tfidf_subject_vectorizer.pkl"))
        instance.body_vectorizer = joblib.load(os.path.join(base_path, "tfidf_body_vectorizer.pkl"))
        logger.info(f"RandomForest model and TF-IDF vectorizers loaded from {base_path}")
        return instance