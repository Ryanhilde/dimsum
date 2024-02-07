import torch
import torch.nn as nn

class ImputationTransformer(nn.Module):
    def __init__(self, input_dim, num_layers=2, nhead=2, dim_feedforward=512, dropout=0.1):
        super(ImputationTransformer, self).__init__()
        self.transformer = nn.Transformer(d_model=input_dim, nhead=nhead, 
                                          num_encoder_layers=num_layers, 
                                          dim_feedforward=dim_feedforward, 
                                          dropout=dropout)
        self.linear_out = nn.Linear(input_dim, 1)

    def forward(self, src):
        output = self.transformer(src, src)
        output = self.linear_out(output)
        return output
