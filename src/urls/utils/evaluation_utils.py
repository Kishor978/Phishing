import os
import logging
from urls.utils.enhanced_classifier import EnhancedURLClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

# Get logger
logger = logging.getLogger(__name__)

def evaluate_enhanced_classifier(model, test_data, char_vocab, gnn_vocab, plot_dir="results/plots/urls"):
    """
    Evaluate the enhanced URL classifier on test data.
    
    Args:
        model: Trained CharCGNN model
        test_data: DataFrame of test data with URLs and labels
        char_vocab: Character vocabulary
        gnn_vocab: GNN vocabulary
        plot_dir: Directory to save plots
        
    Returns:
        dict: Evaluation metrics
    """
    logger.info("Evaluating the enhanced URL classifier...")
    
    # Create plot directory if it doesn't exist
    os.makedirs(plot_dir, exist_ok=True)
    
    # Initialize enhanced classifier
    classifier = EnhancedURLClassifier(model=model, char_vocab=char_vocab, gnn_vocab=gnn_vocab)
    
    # Prepare for evaluation
    urls = test_data['text'].tolist()
    true_labels = test_data['label'].tolist()
    
    # Convert numeric labels to text labels
    true_label_texts = ["Phishing" if label == 1 else "Legitimate" for label in true_labels]
    
    # Predictions
    predictions = []
    confidences = []
    reasons = []
    
    logger.info(f"Evaluating {len(urls)} URLs with enhanced classification...")
    for url in urls:
        prediction, confidence, reason = classifier.classify_url(url)
        predictions.append(prediction)
        confidences.append(confidence)
        reasons.append(reason)
    
    # Calculate metrics
    binary_true = [1 if label == "Phishing" else 0 for label in true_label_texts]
    binary_pred = [1 if pred == "Phishing" else 0 for pred in predictions]
    
    accuracy = accuracy_score(binary_true, binary_pred)
    precision = precision_score(binary_true, binary_pred)
    recall = recall_score(binary_true, binary_pred)
    f1 = f1_score(binary_true, binary_pred)
    confusion = confusion_matrix(binary_true, binary_pred)
    
    # Calculate metrics for whitelist reasons
    whitelist_count = reasons.count("Domain in trusted whitelist")
    rule_based_count = len([r for r in reasons if "rule-based" in r])
    
    # Log results
    logger.info(f"Enhanced Classifier Evaluation Results:")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1 Score: {f1:.4f}")
    logger.info(f"Whitelist Classifications: {whitelist_count} ({whitelist_count/len(urls):.2%})")
    logger.info(f"Rule-Based Classifications: {rule_based_count} ({rule_based_count/len(urls):.2%})")
    
    # Create result dictionary
    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": confusion.tolist(),
        "whitelist_count": whitelist_count,
        "rule_based_count": rule_based_count
    }
    
    # Generate plots
    from urls.utils.plotting_utils import plot_confusion_matrix, plot_roc_curve
    
    # Plot confusion matrix
    plot_confusion_matrix(
        true_labels=binary_true,
        pred_labels=binary_pred,
        classes=["Legitimate", "Phishing"],
        title="Enhanced URL Classifier Confusion Matrix",
        filename=os.path.join(plot_dir, "(Enhanced) confusion_matrix.png")
    )
    
    # Plot ROC curve with confidence scores
    plot_roc_curve(
        y_true=binary_true, 
        y_scores=[conf if pred == "Phishing" else 1-conf for pred, conf in zip(predictions, confidences)],
        title="Enhanced URL Classifier ROC Curve",
        filename=os.path.join(plot_dir, "(Enhanced) roc_curve.png")
    )
    
    return results
