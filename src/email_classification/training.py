import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import copy

from email_classification.email_utils import setup_logging, save_pytorch_model, save_sklearn_model, MODELS_DIR, save_metrics, METRICS_DIR
from email_classification.evaluation import plot_metrics, plot_confusion_matrix, plot_roc_curve # Import plotting utilities
from email_classification.model_config import EPOCHS, PATIENCE, LEARNING_RATE

logger = setup_logging()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_pytorch_model(model, train_loader, val_loader, optimizer, model_type,
                        scheduler=None, epochs=EPOCHS, patience=PATIENCE, learning_rate=LEARNING_RATE):
    """
    Generic training loop for PyTorch models.
    Handles different data formats for single vs dual encoder models.
    """
    # Using BCEWithLogitsLoss for numerical stability with sigmoid activation in the loss
    criterion = nn.BCEWithLogitsLoss()
    
    model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_f1 = 0.0
    patience_counter = 0

    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'train_f1': [], 'val_f1': [],
               'val_preds': [], 'val_probs': [], 'val_labels': []}

    logger.info(f"Starting PyTorch model training on {device}...")
    logger.info(f"Epochs: {epochs}, Patience: {patience}, Learning Rate: {learning_rate}")

    for epoch in range(epochs):
        model.train()
        total_loss, correct_predictions, total_samples = 0, 0, 0
        train_preds, train_labels = [], []

        for batch_idx, batch_data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")):
            optimizer.zero_grad()
            
            logits = None
            labels = None
            
            if model_type == 'single_bilstm': # For BiLSTMClassifier
                text_input, labels = batch_data
                text_input = text_input.to(device)
                labels = labels.to(device)
                logits = model(text_input).squeeze(1) # Squeeze to (batch_size,)
                
            elif model_type in 'dual_bilstm': # For DualEncoderFusion
                subject_input, body_input, labels = batch_data
                subject_input = subject_input.to(device)
                body_input = body_input.to(device)
                labels = labels.to(device)
                logits = model(subject_input, body_input).squeeze(1) # Squeeze to (batch_size,)
            elif model_type== 'dual_transformer': # For  DualEncoderAttentionFusion
                subject_input, body_input, labels = batch_data
                subject_input = subject_input.to(device)
                body_input = body_input.to(device)
                labels = labels.to(device)
                logits = model(subject_input, body_input).squeeze() # Squeeze to (batch_size,)

            else:
                raise ValueError(f"Unknown PyTorch model type: {model_type}")

            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()
            correct_predictions += (preds == labels.int()).sum().item()
            
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.int().cpu().numpy())

        train_loss = total_loss / total_samples if total_samples > 0 else 0
        train_acc = correct_predictions / total_samples if total_samples > 0 else 0
        train_f1 = f1_score(train_labels, train_preds, zero_division=0)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)

        # Validation phase
        model.eval()
        val_total_loss, val_correct_predictions, val_total_samples = 0, 0, 0
        val_preds_epoch, val_probs_epoch, val_labels_epoch = [], [], []

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]")):
                logits = None
                labels = None
                
                if model_type == 'single_bilstm':
                    text_input, labels = batch_data
                    text_input = text_input.to(device)
                    labels = labels.to(device)
                    logits = model(text_input).squeeze(1)
                elif model_type =='dual_bilstm':
                    subject_input, body_input, labels = batch_data
                    subject_input = subject_input.to(device)
                    body_input = body_input.to(device)
                    labels = labels.to(device)
                    logits = model(subject_input, body_input).squeeze(1)
                elif model_type =='dual_transformer':
                    subject_input, body_input, labels = batch_data
                    subject_input = subject_input.to(device)
                    body_input = body_input.to(device)
                    labels = labels.to(device)
                    logits = model(subject_input, body_input).squeeze()
                
                loss = criterion(logits, labels)
                val_total_loss += loss.item() * labels.size(0)
                val_total_samples += labels.size(0)

                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).int()
                val_correct_predictions += (preds == labels.int()).sum().item()

                val_preds_epoch.extend(preds.cpu().numpy())
                val_probs_epoch.extend(probs.cpu().numpy())
                val_labels_epoch.extend(labels.int().cpu().numpy())

        val_loss = val_total_loss / val_total_samples if val_total_samples > 0 else 0
        val_acc = val_correct_predictions / val_total_samples if val_total_samples > 0 else 0
        val_f1 = f1_score(val_labels_epoch, val_preds_epoch, zero_division=0)
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        history['val_preds'] = val_preds_epoch # Store final epoch's predictions
        history['val_probs'] = val_probs_epoch
        history['val_labels'] = val_labels_epoch

        logger.info(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f}, F1={train_f1:.4f} | "
                    f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}")

        if scheduler:
            scheduler.step(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            save_pytorch_model(model, MODELS_DIR, model_name=f"{model_type}_best_model.pt")
            logger.info(f"✅ New best model saved with Val F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            logger.info(f"Patience: {patience_counter}/{patience}")
            if patience_counter >= patience:
                logger.info("⛔ Early stopping triggered.")
                break

    model.load_state_dict(best_model_wts)
    logger.info("Training complete. Loaded best model weights.")
    return model, history

def train_traditional_model(model_wrapper, X_train_subject, X_train_body, y_train, X_val_subject, X_val_body, y_val, model_name="baseline_rf"):
    """
    Trains and evaluates the traditional (RandomForest) model.
    """
    logger.info("Starting Traditional (RandomForest) model training...")
    model_wrapper.fit(X_train_subject, X_train_body, y_train)
    logger.info("Traditional model training complete.")

    logger.info("\n--- Traditional Model Evaluation ---")
    y_pred = model_wrapper.predict(X_val_subject, X_val_body)
    y_prob = model_wrapper.predict_proba(X_val_subject, X_val_body)

    # Calculate metrics
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_val, y_prob)
    
    logger.info(f"Validation Accuracy: {accuracy:.4f}")
    logger.info(f"Validation F1-score: {f1:.4f}")
    logger.info(f"Validation ROC AUC: {roc_auc:.4f}")
    logger.info("Classification Report:\n" + classification_report(y_val, y_pred, zero_division=0))

    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc_score': roc_auc,
        'classification_report': classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    }
    save_metrics(metrics, f"{model_name}_metrics.json")

    # Plotting
    plot_confusion_matrix(y_val, y_pred, title_suffix=f"({model_name.replace('_',' ').title()})")
    plot_roc_curve(y_val, y_prob, title_suffix=f"({model_name.replace('_',' ').title()})")

    # Save model assets
    model_wrapper.save_assets(MODELS_DIR)
    logger.info(f"Traditional model assets saved to {MODELS_DIR}.")
    
    return {'val_labels': y_val.tolist(), 'val_preds': y_pred.tolist(), 'val_probs': y_prob.tolist()}