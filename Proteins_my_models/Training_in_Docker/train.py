from Proteins_my_models.NeuralNetworks.ProteinFunctionAttentionNetwork import ProteinFunctionAttentionNetwork
from Proteins_my_models.Metrics.FocalLoss import FocalLoss
from Proteins_my_models.NeuralNetworks.CustomProteinDataset import ProteinDataset
from torch.utils.data import Dataset, DataLoader
from Proteins_my_models.Metrics.AssessModelPerformance import validate_model
import torch
import pandas as pd

# Setup device (MacBook Air GPU)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

train_m = pd.read_parquet("../processed_data_final_merge/train_master.parquet")
valid_m = pd.read_parquet("../processed_data_final_merge/valid_master.parquet")
test_m  = pd.read_parquet("../processed_data_final_merge/test_master.parquet")

# Find every single unique GO term across the WHOLE project
# This ensures a "Master Checklist"
all_terms = pd.concat([train_m['term'], valid_m['term'], test_m['term']]).unique()
all_go_cols = [f"GO:{t}" if str(t).isdigit() else t for t in all_terms]
all_go_cols.sort()

def process_split(master_df, go_columns):
    embedding_cols = [c for c in master_df.columns if c.startswith("ME:")]
    meta_cols = ["EntryID", "Sequence", "Length"]
    
    # Pivot labels
    labels = (
        master_df[meta_cols + ["term"]]
        .assign(value=1)
        .pivot_table(index=meta_cols, columns="term", values="value", fill_value=0)
        .reset_index()
    )
    labels.columns = [f"GO:{c}" if str(c).isdigit() else c for c in labels.columns]
    
    # Reindex to match the Master Checklist
    # This adds the missing "rare" GO columns as all 0s
    labels = labels.reindex(columns=meta_cols + go_columns, fill_value=0)
    
    # Group embeddings
    embs = master_df.groupby("EntryID")[embedding_cols].mean().reset_index()
    
    # Final Merge
    return pd.merge(labels, embs, on="EntryID")

# Apply to all three
train_df = process_split(train_m, all_go_cols)
valid_df = process_split(valid_m, all_go_cols)
test_df  = process_split(test_m, all_go_cols)

print(f"Train: {train_df.shape} | Val: {valid_df.shape} | Test: {test_df.shape}")

# Initialize model
num_labels = train_df.filter(regex="^GO:").shape[1]
model = ProteinFunctionAttentionNetwork(input_dim=640, output_dim=num_labels).to(device)

# Get data ready for model training
batch_size = 32
train_loader = DataLoader(ProteinDataset(train_df), batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(ProteinDataset(valid_df), batch_size=batch_size)
test_loader  = DataLoader(ProteinDataset(test_df), batch_size=batch_size)

# This gives you the first BATCH
embeddings, targets = next(iter(test_loader))

print(f"Batch Embeddings Shape: {embeddings.shape}") # Should be [32, 640]
print(f"Batch Targets Shape: {targets.shape}")       # Should be [32, 1923]

print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")

# Loss and Optimizer
criterion = FocalLoss(alpha=0.25, gamma=1.5)
optimizer = torch.optim.AdamW(model.parameters(), lr=3.1e-5, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

# Initialize history to capture everything
history = {
    'train_auprc': [], 'val_auprc': [],
    'train_loss': [], 'val_loss': []
}

# Initialize other hyperparameters and metrics for training
epochs = 100
best_val_auprc = 0.0
patience = 10
counter = 0

# Main training loop
for epoch in range(epochs):

    # === Training Phase ===
    model.train()
    running_train_loss = 0.0

    for embeddings, targets in train_loader:
        # Move data & labels to Mac's GPU
        embeddings, targets = embeddings.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    # --- Validation Phase ---
    val_results = validate_model(model, val_loader, device, criterion=criterion)
    
    # Also calculate training metrics for a side-by-side comparison
    train_results = validate_model(model, train_loader, device, criterion=criterion)

    # Update the Scheduler
    scheduler.step(val_results['auprc'])
    current_lr = optimizer.param_groups[0]['lr']

    # Check for Improvement
    if val_results['auprc'] > best_val_auprc:
        best_val_auprc = val_results['auprc']
        torch.save(model.state_dict(), "best_model.pth")
        print(f"🌟 New Best Val AUPRC: {best_val_auprc:.4f} - Model Saved!")
        counter = 0 # Reset early stopping counter
    else:
        counter += 1
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
     # --- RECORDING RESULTS ---
    history['train_loss'].append(running_train_loss / len(train_loader))
    history['val_loss'].append(val_results['loss'].item()) # .item() ensures it's a float
    
    history['train_auprc'].append(train_results['auprc'])
    history['val_auprc'].append(val_results['auprc'])

    print(f"Epoch {epoch+1:02d} | "
          f"Loss (Train/Val): {history['train_loss'][-1]:.4f} / {history['val_loss'][-1]:.4f} | "
          f"AUPRC (Train/Val): {train_results['auprc']:.4f} / {val_results['auprc']:.4f}")

print("hi")