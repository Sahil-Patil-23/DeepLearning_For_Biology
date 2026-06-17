from torch.utils.data import Dataset, DataLoader
import torch

class ProteinDataset(Dataset):
    def __init__(self, df):
        # Extract the embeddings
        self.embeddings = torch.tensor(
            df.filter(regex="^ME:").values, dtype=torch.float32
        )

        # Extract target that our model will try to predict
        self.targets = torch.tensor(
            df.filter(regex="^GO:").values, dtype=torch.float32
        )

    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, index):
        return self.embeddings[index], self.targets[index]