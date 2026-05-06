import torch.nn as nn 
from NeuralNetworks.ResidualLayer import ResidualBlock

class ProteinFunctionResidualNetwork(nn.Module):

    """
    Protein function predictor model that utilizes Residual Blocks.

    This model consists of initial layer using linear transformation, Swish activation, and Batch Norm.
    From here, there are 2 Residual Blocks and a final linear layer that classifies the proteins.
    """

    def __init__(self, input_dim=640, output_dim=1923):
        super().__init__()

        # Initial layer is standard layer with batch normalization
        self.initial_layer = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=1024),
            nn.SiLU(),
            nn.BatchNorm1d(1024)
        )

        # Residual layers
        self.res_layer1 = ResidualBlock(size=1024, dropout=0.25)
        self.res_layer2 = ResidualBlock(size=1024, dropout=0.25)

        # Final layer that classifies the proteins
        self.classifier = nn.Linear(in_features=1024, out_features=output_dim)

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
        return self.classifier(x)