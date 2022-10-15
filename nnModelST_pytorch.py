import torch
from torch import nn
from torch.nn import functional as F
from gcnModelST_pytorch import GCN_layer
import torch
from gcnModelST_pytorch import GCN_layer
from GCN_attention import attention

class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(60, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.Dropout(0.25),
            nn.BatchNorm2d(64)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(4, 4), stride=(4, 4)),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.Dropout(0.25),
            nn.BatchNorm2d(64)
        )
        # self.layer3 = nn.Sequential(
        #     nn.Linear(1024, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ELU(inplace=True),
        #     nn.Dropout(0.25),
        # )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(4, 4), stride=(4, 4)),
            nn.BatchNorm2d(64),
            nn.ELU(inplace=True),
            nn.Dropout(0.25),
            nn.BatchNorm2d(64)
        )

        self.layer4 = nn.Sequential(
            nn.Linear(4 * 64, 64),
            # nn.BatchNorm1d(64),
            nn.ELU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(64, 2),
            nn.Softmax(dim=1)
            )

    def forward(self, img):
        x1 = self.layer1(img)
        x2 = self.layer2(x1)
        xb = self.layer3(x2)
        # print(xb.shape)
        x3 = xb.contiguous().view(-1, 4*64)
        x4 = self.layer4(x3)
        #4x = self.layer4(x)
        return x4

