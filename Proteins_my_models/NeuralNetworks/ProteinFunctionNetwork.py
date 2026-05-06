import torch.nn as nn 

class ProteinFunctionNetwork(nn.Module):

    """
    Protein function predictor model that utilizes linear layers, batch norm, swish activation function, and dropout.

    There are 4 layers in total, but the final one is the classification layer that predicts protein function(s).
    """

    def __init__(self, input_dim=640, output_dim=1923):
        super().__init__()
        self.neural_network = nn.Sequential(

            # Layer 1
            nn.Linear(in_features=input_dim, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(),
            nn.Dropout(0.15),

            # Layer 2
            nn.Linear(in_features=1024, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(),
            nn.Dropout(0.15),

            # Layer 3
            nn.Linear(in_features=1024, out_features=1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(),
            nn.Dropout(0.15),

            # Layer 4 (Classification Layer that predicts protein function(s))
            nn.Linear(in_features=1024, out_features=output_dim)

        )

    def forward(self, x):

        """
        Performs the forward pass of the network.

        Args:
            x (Tensor): Input data with shape (batch_size, input_features).

        Returns:
            Tensor: Output vector of predicted protein sequences.
        """

        return self.neural_network(x)