import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=33),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=17),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=9),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=256, out_channels=521, kernel_size=5),
            nn.LeakyReLU(0.2)
        )
        self.block2 = nn.Linear(521 * 4 * 4, 1)

        return torch.cat(self.block1, self.block2, 0)

    def forward(self, input):
        output = self.block1(input)
        output = output.view(-1, 521 * 4 * 4)
        output = self.block2(output)

        return output
