import torch.optim as optim
import torch
import numpy as np
from solution.model import ConvNet
from solution.learning import prepare_loaders, validate, run_epoch
from solution.preprocessing import MainTransform, NumberToTensor


train_loader, val_loader, test_loader = prepare_loaders('ref_datasets/ref_sc_v2_17000.csv',
                                                        MainTransform(), NumberToTensor())

model_name = 'main_model'
model = ConvNet()
model.load_state_dict(torch.load(f'model_states/{model_name}.pt'))


labels = []
pred_labels = []
# samples = []
model.eval()
with torch.no_grad():
    for k, (x, y) in enumerate(test_loader):
        y_hat = torch.sigmoid(model(x))
        pred_labels.append(np.round(y_hat.numpy()))
        labels.append(y.cpu().numpy())
        # samples.append(x.cpu().numpy())

labels = np.concatenate(labels).flatten()
pred_labels = np.concatenate(pred_labels).flatten()
# samples = np.concatenate(samples)

from sklearn.metrics import classification_report
print(classification_report(labels, pred_labels))