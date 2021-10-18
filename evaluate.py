import torch.optim as optim
import torch
import numpy as np
from solution.model import ConvNet
from solution.learning import prepare_loaders, validate, run_epoch
from solution.preprocessing import MainTransform, NumberToTensor


train_loader, val_loader, test_loader = prepare_loaders('ref_datasets/ref_sc_v2_17000.csv',
                                                        MainTransform(), NumberToTensor())

def collect_pred(loader, model_name='main_model'):
    model = ConvNet()
    model.load_state_dict(torch.load(f'model_states/{model_name}.pt'))

    labels = []
    preds = []

    model.eval()
    with torch.no_grad():
        for k, (x, y) in enumerate(loader):
            y_hat = torch.sigmoid(model(x))
            preds.append(y_hat.numpy())
            labels.append(y.cpu().numpy())

    labels = np.concatenate(labels).flatten()
    preds = np.concatenate(preds).flatten()
    return labels, preds

# learning curve
model_names = ['convnet_17000_1ep', 'convnet_17000_3ep', 'convnet_17000_10ep', 'main_model']
epochs = [1, 3, 10, 30]

val_labels = []
val_preds = []
for model_name in model_names:
    print(f'collecting val. preds for {model_name}')
    labels, preds = collect_pred(val_loader, model_name)
    val_labels.append(labels)
    val_preds.append(preds)

print(f'collecting test. preds for main_model')
test_labels, test_pred = collect_pred(test_loader, 'main_model')

from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef, precision_recall_curve
import pylab as plt

mcc_valid = [matthews_corrcoef(l, np.round(p)) for l, p in zip(val_labels, val_preds)]
presicion, recall, th = precision_recall_curve(test_labels, test_pred)

th05_ind = np.argmin(np.abs(th-0.5))

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
axes[0].plot(epochs, mcc_valid, 'o--', label='validation')
axes[0].plot(epochs[-1], matthews_corrcoef(test_labels, np.round(test_pred)), 'o', label='test')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MCC')
axes[0].set_title('A. Learning curve')
axes[0].legend()
axes[0].set_box_aspect(1)

axes[1].plot(presicion, recall, 'C1', label='all thresholds')
axes[1].plot(presicion[th05_ind], recall[th05_ind], 'oC1', label='threshold=0.5')
axes[1].set_xlabel('Precision')
axes[1].set_ylabel('Recall')
axes[1].set_title('B. Precision-recall curve')
axes[1].legend()
axes[1].set_box_aspect(1)

plt.subplots_adjust(wspace=0.3)

plt.savefig('images/learning_and_pr_curves.png', dpi=150)

# print(matthews_corrcoef(labels, np.round(preds)))

# print(classification_report(labels, pred_labels))
