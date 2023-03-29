from torch import nn

class layer_4(nn.Module):
    def __init__(self, feature_num):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(feature_num, 256),
            nn.Sigmoid(),
            nn.Linear(256, 512),
            nn.Sigmoid(),
            nn.Linear(512, 16),
            nn.Sigmoid(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        out_put = self.linear_relu_stack(x)
        return out_put