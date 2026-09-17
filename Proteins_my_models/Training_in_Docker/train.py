from Proteins_my_models.NeuralNetworks.ProteinFunctionAttentionNetwork import ProteinFunctionAttentionNetwork
from Proteins_my_models.Metrics.FocalLoss import FocalLoss
from Proteins_my_models.NeuralNetworks.CustomProteinDataset import ProteinDataset
from torch.utils.data import Dataset, DataLoader
from Proteins_my_models.Metrics.AssessModelPerformance import compute_metrics, batch_metric_sums
import torch
import pandas as pd

def process_split(master_df, go_columns):
    '''
    
    '''
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

def main():
    # Setup device (Nvidia RTX5070)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        # Allows PyTorch to use faster, but slightly lower precision algorithms
        torch.set_float32_matmul_precision('high')

    train_m = pd.read_parquet("../processed_data_final_merge/train_master.parquet")
    valid_m = pd.read_parquet("../processed_data_final_merge/valid_master.parquet")
    test_m  = pd.read_parquet("../processed_data_final_merge/test_master.parquet")

    # Find every single unique GO term across the WHOLE project
    # This ensures a "Master Checklist"
    all_terms = pd.concat([train_m['term'], valid_m['term'], test_m['term']]).unique()
    all_go_cols = [f"GO:{t}" if str(t).isdigit() else t for t in all_terms]
    all_go_cols.sort()


    # Apply to all three
    train_df = process_split(train_m, all_go_cols)
    valid_df = process_split(valid_m, all_go_cols)
    test_df  = process_split(test_m, all_go_cols)

    print(f"Train: {train_df.shape} | Val: {valid_df.shape} | Test: {test_df.shape}")

    # Initialize model
    num_labels = train_df.filter(regex="^GO:").shape[1]
    model = ProteinFunctionAttentionNetwork(input_dim=640, output_dim=num_labels).to(device)

    # Ideally should yield a speedup
    # try:
    #     model = torch.compile(model)
    #     print("Torch compile of model successful!")
    # except Exception as e:
    #     print(f"Torch compile not successful: {e}")

    # Get data ready for model training
    batch_size = 256
    train_loader = DataLoader(ProteinDataset(train_df), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(ProteinDataset(valid_df), batch_size=batch_size, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(ProteinDataset(test_df), batch_size=batch_size)

    # This gives you the first BATCH
    embeddings, targets = next(iter(test_loader))

    print(f"Batch Embeddings Shape: {embeddings.shape}") # Should be [256, 640]
    print(f"Batch Targets Shape: {targets.shape}")       # Should be [256, 1923]

    print(f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")

    # Loss and Optimizer
    criterion = FocalLoss(alpha=0.25, gamma=1.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    # Initialize Gradient Scaler for AMP
    scaler = torch.amp.GradScaler('cuda')

    # Initialize history to capture everything
    history = {
        'train_auprc': [], 'val_auprc': [],
        'train_loss': [], 'val_loss': []
    }

    # Initialize other hyperparameters and metrics for training
    epochs = 200
    best_val_auprc = 0.0
    patience = 10
    counter = 0

    # Main training loop
    for epoch in range(epochs):

        # === Training Phase ===
        # model.train()
        # all_train_batch_metrics = []
        # running_train_loss = 0.0

        # for embeddings, targets in train_loader:
        #     # Move data & labels to GPU
        #     embeddings, targets = embeddings.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        #     optimizer.zero_grad()

        #     with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        #         outputs = model(embeddings)
        #         loss = criterion(outputs, targets)
        #     scaler.scale(loss).backward()
        #     scaler.step(optimizer)
        #     scaler.update()
        #     running_train_loss += loss.item()

        #     with torch.no_grad():
        #         # Apply Sigmoid to get probabilities (0 to 1)
        #         probs = torch.sigmoid(outputs).detach().cpu().float().numpy()
        #         targets_np = targets.cpu().numpy()
    
        #         # Calculate metrics for each protein in the batch (Author's logic)
        #         for i in range(len(targets_np)):
        #             m = compute_metrics(targets_np[i], probs[i])
        #             all_train_batch_metrics.append(m)

        # train_summary = pd.DataFrame(all_train_batch_metrics).mean()
        # train_results = train_summary.to_dict()
        # train_results['loss'] = running_train_loss / len(train_loader)
        model.train()
        running_train_loss = 0.0
        train_metric_sums = {"auprc": 0.0, "auroc": 0.0}
        train_total_valid = 0

        for embeddings, targets in train_loader:
            embeddings, targets = embeddings.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            optimizer.zero_grad()

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(embeddings)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_train_loss += loss.item()

            with torch.no_grad():
                probs = torch.sigmoid(outputs).detach().cpu().float().numpy()
                targets_np = targets.cpu().numpy()
                sums, n = batch_metric_sums(targets_np, probs)
                if sums is not None:
                    for k in train_metric_sums:
                        train_metric_sums[k] += sums[k]
                    train_total_valid += n

        train_results = {k: (v / train_total_valid if train_total_valid else 0.0)
                        for k, v in train_metric_sums.items()}
        train_results['loss'] = running_train_loss / len(train_loader)

        # === Validation Phase ===
        # model.eval()
        # all_val_batch_metrics = []
        # running_val_loss = 0.0

        # with torch.no_grad():
        #     for embeddings, targets in val_loader:
        #         embeddings, targets = embeddings.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        #         with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        #             outputs = model(embeddings)
        #             loss = criterion(outputs, targets)
        #         running_val_loss += loss.item()

        #         # Apply Sigmoid to get probabilities (0 to 1)
        #         probs = torch.sigmoid(outputs).cpu().float().numpy()
        #         targets_np = targets.cpu().numpy()
    
        #         # Calculate metrics for each protein in the batch (Author's logic)
        #         for i in range(len(targets_np)):
        #             m = compute_metrics(targets_np[i], probs[i])
        #             all_val_batch_metrics.append(m)

        # val_summary = pd.DataFrame(all_val_batch_metrics).mean()
        # val_results = val_summary.to_dict()
        # val_results['loss'] = running_val_loss / len(val_loader)
        model.eval()
        running_val_loss = 0.0
        val_metric_sums = {"auprc": 0.0, "auroc": 0.0}
        val_total_valid = 0

        with torch.no_grad():
            for embeddings, targets in val_loader:
                embeddings, targets = embeddings.to(device, non_blocking=True), targets.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(embeddings)
                    loss = criterion(outputs, targets)
                running_val_loss += loss.item()

                probs = torch.sigmoid(outputs).cpu().float().numpy()
                targets_np = targets.cpu().numpy()
                sums, n = batch_metric_sums(targets_np, probs)
                if sums is not None:
                    for k in val_metric_sums:
                        val_metric_sums[k] += sums[k]
                    val_total_valid += n

        val_results = {k: (v / val_total_valid if val_total_valid else 0.0)
                    for k, v in val_metric_sums.items()}
        val_results['loss'] = running_val_loss / len(val_loader)
                

        # Update the Scheduler
        scheduler.step(val_results['auprc'])
        current_lr = optimizer.param_groups[0]['lr']

         # --- RECORDING RESULTS ---
        history['train_loss'].append(running_train_loss / len(train_loader))
        history['val_loss'].append(running_val_loss / len(val_loader)) # .item() ensures it's a float

        history['train_auprc'].append(train_results['auprc'])
        history['val_auprc'].append(val_results['auprc'])

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

        # Output the metrics for each epoch
        print(f"Epoch {epoch+1:02d} | "
            f"LR: {current_lr} | "  
            f"Loss (Train/Val): {history['train_loss'][-1]:.4f}/{history['val_loss'][-1]:.4f} | "
            f"AUPRC (Train/Val): {train_results['auprc']:.4f}/{val_results['auprc']:.4f}")
    
    

if __name__ == "__main__":
    main()