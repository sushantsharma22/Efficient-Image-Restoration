import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_features=64):
        super(UNet, self).__init__()
        self.encoder1 = self.conv_block(in_channels, base_features)
        self.encoder2 = self.conv_block(base_features, base_features * 2)
        self.decoder1 = self.conv_block(base_features * 2, base_features)
        self.decoder2 = nn.Conv2d(base_features, out_channels, kernel_size=1)

    def conv_block(self, in_feat, out_feat):
        return nn.Sequential(
            nn.Conv2d(in_feat, out_feat, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_feat, out_feat, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(nn.MaxPool2d(2)(enc1))
        dec1 = self.decoder1(nn.Upsample(scale_factor=2)(enc2))
        out = self.decoder2(dec1 + enc1)  # Skip connection
        return out
