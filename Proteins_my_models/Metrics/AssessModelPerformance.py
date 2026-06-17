from sklearn import metrics
import numpy as np
import torch
import pandas as pd

def compute_metrics(targets, probs, thresh=0.5):
    """
    Translation of the author's compute_metrics into PyTorch-friendly logic.
    targets: numpy array of 0s and 1s
    probs: numpy array of predicted probabilities (0.0 to 1.0)
    """
    # If the protein has no labels (rare but possible), return zeros
    if np.sum(targets) == 0:
        return {m: 0.0 for m in ["accuracy", "recall", "precision", "auprc", "auroc"]}

    preds = (probs >= thresh).astype(int)
    
    return {
        "accuracy": metrics.accuracy_score(targets, preds),
        "recall": metrics.recall_score(targets, preds, zero_division=0),
        "precision": metrics.precision_score(targets, preds, zero_division=0),
        "auprc": metrics.average_precision_score(targets, probs),
        # Using try/except because AUROC fails if a sample has only one class
        "auroc": metrics.roc_auc_score(targets, probs) if len(np.unique(targets)) > 1 else 0.5
    }

def validate_model(model, loader, device, criterion=None):
    model.eval()
    all_batch_metrics = []
    total_loss = 0.0
    
    with torch.no_grad():
        for embeddings, targets in loader:
            embeddings, targets = embeddings.to(device), targets.to(device)
            
            # Get model predictions (logits)
            logits = model(embeddings)

            # Calculate Loss
            if criterion != None:
                loss = criterion(logits, targets)
                total_loss += loss

            # Apply Sigmoid to get probabilities (0 to 1)
            probs = torch.sigmoid(logits).cpu().numpy()
            targets_np = targets.cpu().numpy()

            # Calculate metrics for each protein in the batch (Author's logic)
            for i in range(len(targets_np)):
                m = compute_metrics(targets_np[i], probs[i])
                all_batch_metrics.append(m)
                
    # Average the metrics across all proteins
    summary_df = pd.DataFrame(all_batch_metrics).mean()
    results = summary_df.to_dict()

    if criterion != None:
        results['loss'] = total_loss / len(loader)
        
    return results