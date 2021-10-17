import torch
from torch import nn

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=3, kernel_size=(32, 16))
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(16, 8))
        self.fc = nn.Linear(3*13*4, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = torch.relu(out)
        out = torch.max_pool2d(out, (8, 4))
        out = self.conv2(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    from solution.data import SCDataset
    from solution.preprocessing import MainTransform, NumberToTensor
    data = SCDataset('ref_datasets/ref_sc_v2_17000.csv',
                     transform=MainTransform(), transform_label=NumberToTensor())
    waveform, label = data[0]

    model = ConvNet()
    out = model(waveform.unsqueeze(0))