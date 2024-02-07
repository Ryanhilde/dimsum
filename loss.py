import torch
import torch.nn.functional as F

def composite_loss(output, target, alpha=1.0, beta=1.0, gamma=1.0):
    mae_loss = torch.mean(torch.abs(target['value'] - output))
    bucket_loss = F.cross_entropy(output, target['bucket_label'])
    total_loss = F.mse_loss(output.sum(dim=1), target['total'])
    
    total_composite_loss = alpha * mae_loss + beta * bucket_loss + gamma * total_loss
    return total_composite_loss
