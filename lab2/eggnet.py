import dataloader
import torch
import torch.nn as nn
class EEGnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.firstconv = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False),
                nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
                )
        self.depthwiseconv = nn.Sequential(
                nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False),
                nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ELU(alpha=1.0),
                nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0),
                nn.Dropout(p=0.25)
                )

    def forward(self, x):
        firstconvout = self.firstconv(x)
        depthwiseconvout = self.depthwiseconv(firstconvout)

        return depthwiseconvout


if __name__ == '__main__':
    dataloader.read_bci_data()

    model = EEGnet()
    print(model)
 
