import numpy as np
import torch.optim as optim
import torch

from solution.model import ConvNet
from solution.learning import prepare_loaders, validate, run_epoch
from solution.preprocessing import MainTransform, NumberToTensor


ref_dataset = 'ref_datasets/ref_sc_v2_17000.csv'
model_name = 'convnet_17000'

model = ConvNet()
loss_fn = torch.nn.BCEWithLogitsLoss()

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(device)

model.to(device=device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_loader, val_loader, test_loader = prepare_loaders(ref_dataset, MainTransform(), NumberToTensor())

train_scores = []
val_scores = []
val_loss, val_score = validate(model, val_loader, device, loss_fn)
print(f'EPOCH -1 VALIDATION: '
      f'loss {val_loss.mean():.5f}+-{val_loss.std():.5f} || '
      f'acc  {val_score.mean():.5f}+-{val_score.std():.5f}')

n_epochs = 30
for epoch in range(1, n_epochs + 1):
    train_score = run_epoch(epoch, train_loader, model, loss_fn, optimizer, device)
    val_loss, val_score = validate(model, val_loader, device, loss_fn)
    print(f'EPOCH {epoch} VALIDATION: '
          f'loss {val_loss.mean():.5f}+-{val_loss.std():.5f} || '
          f'acc  {val_score.mean():.5f}+-{val_score.std():.5f}')
    train_scores.append(train_score)
    val_scores.append(val_score)

val_loss, val_score = validate(model, test_loader, device, loss_fn)
print(f'TEST VALIDATION: '
      f'loss {val_loss.mean():.5f}+-{val_loss.std():.5f} || '
      f'acc  {val_score.mean():.5f}+-{val_score.std():.5f}')


torch.save(model.state_dict(), f'model_states/{model_name}.pt')




