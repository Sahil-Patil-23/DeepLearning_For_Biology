import torch.nn as nn

class ResidualBlock(nn.Module):

    """
    Residual block in which output of a layer is added back into input.

    Applies linear transformation to incoming data, normalizes output of linear layer to stabilize training,
    applies Swish activation function, and finally utilizes dropout to prevent overfitting.
    """

    def __init__(self, size, dropout=0.25):
        super().__init__()

        """
        Initializes the Residual Block with specified size & dropout.

        Args:
            - size (int): Size of feature dimension
            - dropout (float): Proportion of neurons that get zeroed with this Residual Block. Default is
                               0.25, but can be overridden.
        """

        self.res_block = nn.Sequential(
            nn.Linear(in_features=size, out_features=size),
            nn.BatchNorm1d(size),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        """
        Performs the forward pass of the model.

        Args:
            - x (torch.tensor): Input tensor of shape (batch_size, input_dim)
        """

        return x + self.res_block(x)