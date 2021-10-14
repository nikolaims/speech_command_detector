import numpy as np
import torch.optim as optim
import torch

from solution.model import ConvNet
from solution.learning import prepare_loaders, validate, run_epoch
from solution.preprocessing import MainTransform, NumberToTensor




model = ConvNet()
loss_fn = torch.nn.BCEWithLogitsLoss()

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
print(device)

model.to(device=device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

train_loader, val_loader, test_loader = prepare_loaders('/Users/kolai/Data/speech_commands_v0.01/ref_full_10000.csv',
                                                        MainTransform(), NumberToTensor())

train_scores = []
val_scores = []
val_loss, val_score = validate(model, val_loader, device, loss_fn)
print(f'EPOCH -1 VALIDATION: '
      f'loss {val_loss.mean():.5f}+-{val_loss.std():.5f} || '
      f'acc  {val_score.mean():.5f}+-{val_score.std():.5f}')

n_epochs = 10


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

model_name = 'small_1000'
torch.save(model.state_dict(), f'results/{model_name}.pt')




