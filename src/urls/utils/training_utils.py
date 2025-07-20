import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_curve, auc
import copy # For deepcopying model state_dict for best model saving

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(model, train_loader, val_loader, optimizer, criterion, scheduler=None, epochs=30, patience=5, model_save_path="best_model.pt", model_type="gnn"):
    """
    Generic training loop for PyTorch models (GNN or CharCNN).
    Includes early stopping and saving of the best model.

    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        criterion (torch.nn.Module): Loss function (e.g., BCELoss).
        scheduler (torch.optim.lr_scheduler, optional): Learning rate scheduler. Defaults to None.
        epochs (int, optional): Number of training epochs. Defaults to 30.
        patience (int, optional): Early stopping patience. Defaults to 5.
        model_save_path (str, optional): Path to save the best model. Defaults to "best_model.pt".
        model_type (str, optional): Type of model ("gnn", "charcnn", "fusion") to determine data handling.
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_f1 = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_preds': [], 'val_probs': [], 'val_labels': []}

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        train_preds, train_labels = [], []

        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            optimizer.zero_grad()
            out = None
            labels = None
            num_samples_in_batch = 0

            if model_type == "gnn":
                batch_data = batch_data.to(device)
                out = model(batch_data).squeeze() # Ensure output is 1D
                labels = batch_data.y
                num_samples_in_batch = batch_data.num_graphs
            elif model_type == "charcnn":
                x, y = batch_data
                x, y = x.to(device), y.to(device)
                out = model(x).squeeze() # Ensure output is 1D
                labels = y
                num_samples_in_batch = y.size(0)
            elif model_type == "fusion":
                char_input, graph_data = batch_data
                char_input = char_input.to(device)
                graph_data = graph_data.to(device)
                out = model(char_input, graph_data).squeeze() # Ensure output is 1D
                labels = graph_data.y
                num_samples_in_batch = graph_data.num_graphs
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * num_samples_in_batch
            preds = (out > 0.5).int()
            correct += (preds == labels.int()).sum().item()
            total += num_samples_in_batch

            train_preds.extend(preds.cpu().numpy().tolist())
            train_labels.extend(labels.int().cpu().numpy().tolist())


        train_loss = total_loss / total if total > 0 else 0
        train_acc = correct / total if total > 0 else 0
        train_f1 = f1_score(train_labels, train_preds) if total > 0 else 0 # Calculate train F1


        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)

        # Validation
        model.eval()
        val_loss, val_preds, val_probs, val_labels = 0, [], [], []
        with torch.no_grad():
            val_total_samples = 0
            for batch_data in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                out = None
                labels = None
                num_samples_in_batch = 0

                if model_type == "gnn":
                    batch_data = batch_data.to(device)
                    out = model(batch_data).squeeze()
                    labels = batch_data.y
                    num_samples_in_batch = batch_data.num_graphs
                elif model_type == "charcnn":
                    x, y = batch_data
                    x, y = x.to(device), y.to(device)
                    out = model(x).squeeze()
                    labels = y
                    num_samples_in_batch = y.size(0)
                elif model_type == "fusion":
                    char_input, graph_data = batch_data
                    char_input = char_input.to(device)
                    graph_data = graph_data.to(device)
                    out = model(char_input, graph_data).squeeze()
                    labels = graph_data.y
                    num_samples_in_batch = graph_data.num_graphs
                else:
                    raise ValueError(f"Unknown model_type: {model_type}")

                loss = criterion(out, labels)
                val_loss += loss.item() * num_samples_in_batch
                probs = out.cpu().numpy().tolist()
                preds = (out > 0.5).int().cpu().numpy().tolist()
                val_probs.extend(probs)
                val_preds.extend(preds)
                val_labels.extend(labels.int().cpu().numpy().tolist())
                val_total_samples += num_samples_in_batch

        val_loss = val_loss / val_total_samples if val_total_samples > 0 else 0
        val_acc = accuracy_score(val_labels, val_preds) if val_total_samples > 0 else 0
        val_f1 = f1_score(val_labels, val_preds) if val_total_samples > 0 else 0

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        # Store latest validation predictions for plotting
        history['val_preds'] = val_preds
        history['val_probs'] = val_probs
        history['val_labels'] = val_labels


        print(f"\nEpoch {epoch+1}: Train Loss={train_loss:.4f}, Acc={train_acc:.4f}, F1={train_f1:.4f} | Val Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}")

        if scheduler:
            scheduler.step(val_f1) # Assuming scheduler uses validation F1 to adjust LR

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ New best model saved with F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("\n⛔ Early stopping triggered.")
                break

    model.load_state_dict(best_model_wts)
    return model, history
