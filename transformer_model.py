import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class EnhancedTransformerModel(nn.Module):
    def __init__(self, input_dim, seq_len, output_dim, num_layers=2, nhead=2, dim_feedforward=512, dropout=0.1):
        super(EnhancedTransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model=input_dim, dropout=dropout, max_len=seq_len)
        transformer_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(transformer_layer, num_layers=num_layers)
        self.linear_out = nn.Linear(input_dim * seq_len, output_dim)

    def forward(self, src):
        src = src.permute(1, 0, 2)  # Shape: (seq_len, batch_size, input_dim)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.permute(1, 0, 2)  # Shape: (batch_size, seq_len, input_dim)
        flattened = output.view(output.size(0), -1)  # Flatten
        output = self.linear_out(flattened)
        return output.view(-1, src.size(0), output_dim)  # Reshape to match expected output
