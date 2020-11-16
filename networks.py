import torch.nn as nn


class ConvUpscaleDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=16,
                      kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=4,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=4),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16,
                      kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=16, out_channels=32,
                      kernel_size=5, padding=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=32, out_channels=1,
                      kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=1),
            nn.Sigmoid()
        )

    def forward(self, t):
        t = self.encoder(t)
        t = self.decoder(t)

        return t


class ConvTransposeDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32,
                      kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=16,
                      kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=4,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=4),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=4, out_channels=16,
                               kernel_size=2, padding=0, stride=2, bias=False),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=32,
                               kernel_size=4, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=1,
                      kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(num_features=1),
            nn.Sigmoid()
        )

    def forward(self, t):
        t = self.encoder(t)
        t = self.decoder(t)

        return t
