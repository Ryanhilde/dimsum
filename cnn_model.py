import torch
import torch.nn as nn

class EnhancedCNNModel(nn.Module):
    def __init__(self, input_dim, seq_len, output_dim):
        super(EnhancedCNNModel, self).__init__()
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.linear_out = nn.Linear(128 * (seq_len // 4), output_dim)  # Adjusted for two pooling layers

    def forward(self, src):
        src = src.transpose(1, 2)  # Shape: (batch_size, input_dim, seq_len)
        cnn_output = self.cnn_layers(src)
        flattened = cnn_output.view(cnn_output.size(0), -1)  # Flatten
        output = self.linear_out(flattened)
        return output.view(-1, src.size(2), output_dim)  # Reshape to match expected output
