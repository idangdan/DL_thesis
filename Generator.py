class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.block1 = nn.Linear(32, 521 * 4 * 4)

        self.block2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=521, out_channels=256, kernel_size=5),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=9),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=17),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=33),
            nn.LeakyReLU(0.2),
        )
        return torch.cat(self.block1, self.block2, 0)

    def forward(self, input):
        output = self.block1(input)
        output = output.view(-1, 521, 4, 4)
        output = self.block2(output)
        # return output.view(-1, CHANNELS, IM_SIZE, IM_SIZE)

#netG = Generator().to(device)