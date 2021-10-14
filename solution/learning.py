import numpy as np
import torch

from torch.utils.data import DataLoader
from solution.data import SCDataset


def run_epoch(epoch, train_loader, model, loss_fn, optimizer, device):
    loss_acc = 0
    for batch, (x, y) in enumerate(train_loader):
        x = x.to(device=device)
        y = y.to(device=device)
        y_hat = model(x)
        loss = loss_fn(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_acc = loss_acc * 0.5 + loss.item() * 0.5

        print('Epoch {}, Batch {}, Loss {}'.format(epoch, batch, loss_acc))

    return loss_acc


def accuracy(pred, labels):
    pred_labels = torch.round(torch.sigmoid(pred))
    acc = (pred_labels == labels).sum().float()/labels.shape[0]
    return acc

def validate(model, loader, device, loss_fn):
    # очень медленно работет
    losses = []
    acc = []
    with torch.no_grad():
        for k, (x, y) in enumerate(loader):
            y_hat = model(x.to(device=device)).cpu()
            losses.append(loss_fn(y_hat, y).item())
            acc.append(accuracy(y_hat, y).item())
    return np.array(losses), np.array(acc)


def prepare_loaders(ref_dataset_csv_path, transform, transform_label):
    train_dataset = SCDataset(ref_dataset_csv_path, subset='train',
                              transform=transform, transform_label=transform_label)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    valid_dataset = SCDataset(ref_dataset_csv_path, subset='valid',
                              transform=transform, transform_label=transform_label)
    valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=True)

    test_dataset = SCDataset(ref_dataset_csv_path, subset='test',
                             transform=transform, transform_label=transform_label)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    return train_loader, valid_loader, test_loader