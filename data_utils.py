import torch
from torch.utils.data import DataLoader, TensorDataset, random_split

def normalize_data(data):
    return (data - data.mean()) / data.std()

def prepare_data(dataset, train_split=0.8):
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

def get_dataloaders(train_dataset, val_dataset, batch_size=32):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    return train_loader, val_loader
