import torch
import numpy as np

from solution.preprocessing import MainTransform
from solution.data import SAMPLING_RATE, SAMPLES_LEN

OUT_FORMATS = ['raw', 'proba', 'label']


class InferModel:
    def __init__(self, model_cls, model_state_path, transform=None, out_format=None):
        self.transform = transform or MainTransform()
        self.out_format = out_format or 'proba'
        assert self.out_format in OUT_FORMATS, f'Wrong out format, got {self.out_format}, expected on of {OUT_FORMATS}'
        self.model = model_cls()
        self.model.load_state_dict(torch.load(model_state_path))
        self.model.eval()

    def __call__(self, sample):
        with torch.no_grad():
            out = self.model(self.transform(sample.reshape(1, -1)))
            if self.out_format in ['proba', 'label']:
                out = torch.sigmoid(out).item()
                if self.out_format == 'label':
                    out = int(out >= 0.5)
            else:
                out = out.item()
        return out

    def continuous(self, x, hop_ms):
        predictions = np.zeros(len(x))
        hop = int(hop_ms / 1000 * SAMPLING_RATE)
        n_predictions_per_time = (SAMPLES_LEN / hop)
        for start in range(0, len(x) - SAMPLES_LEN, hop):
            x_slice = x[start:start + SAMPLES_LEN]
            predictions[start:start + SAMPLES_LEN] += self(x_slice)
        return predictions / n_predictions_per_time


if __name__ == '__main__':
    from solution.model import ConvNet


    model_name = 'small_1000'
    model_state_path = f'results/{model_name}.pt'
    infer_model = InferModel(ConvNet, model_state_path, out_format='proba')

    print(infer_model(np.random.randn(SAMPLES_LEN)))
    print(infer_model.continuous(np.random.randn(SAMPLES_LEN*2), hop_ms=100))
