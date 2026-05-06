import torch
import torch.nn as nn
from NeuralNetworks.ResidualLayer import ResidualBlock


class LabelAttentionHeader(nn.Module):
    def __init__(self, feature_dim, num_labels):
        super().__init__()
        # 1. Create a "Prototype" for every GO term
        # This is a bank of 1923 vectors, each 1024-dim
        self.label_embeddings = nn.Parameter(torch.empty(num_labels, feature_dim))
        nn.init.xavier_uniform_(self.label_embeddings)
        
        # 2. Scaling factor (like in Transformer self-attention)
        # Helps keep the gradients stable as dimensions grow
        self.scale = feature_dim ** -0.5

    def forward(self, x):
        # x: [Batch, 1024] 
        # label_embeddings: [1923, 1024]
        
        # We calculate the dot-product similarity (cosine-adjacent)
        # [Batch, 1024] @ [1024, 1923] -> [Batch, 1923]
        logits = torch.matmul(x, self.label_embeddings.t()) * self.scale
        return logits

class ProteinFunctionAttentionNetwork(nn.Module):

    """
    Protein function predictor model that utilizes Residual Blocks & Attention mechanism.

    This model consists of initial layer using linear transformation, Swish activation, and Batch Norm.
    From here, there are 2 Residual Blocks.
    """

    def __init__(self, input_dim=640, hidden_dim=2048, output_dim=1923):
        super().__init__()

        # Initial layer is standard layer with batch normalization (Same as 'ProteinFunctionResidualNetwork')
        self.initial_layer = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.SiLU(),
            nn.BatchNorm1d(hidden_dim)
        )

        self.res_layer1 = ResidualBlock(size=hidden_dim, dropout=0.15)
        self.res_layer2 = ResidualBlock(size=hidden_dim, dropout=0.15)

        # Added a non-linear projection before the Attention Header
        self.feature_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.classifier = LabelAttentionHeader(feature_dim=hidden_dim, num_labels=output_dim)

    def forward(self, x):
        """
        Performs the forward pass of the network.

        Args:
            x (Tensor): Input data with shape (batch_size, input_features).

        Returns:
            Tensor: Output vector of predicted protein sequences.
        """
        x = self.initial_layer(x)
        x = self.res_layer1(x)
        x = self.res_layer2(x)
        x = self.feature_projection(x)
        return self.classifier(x)