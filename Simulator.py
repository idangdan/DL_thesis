import torch as torch
import torch.nn as nn


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class Simulator(nn.Module):
    def __init__(self):
        super(Simulator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=33),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=17),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=9),
            nn.LeakyReLU(0.2),
            View(-1))
        #DONE: do pytorch flatten and concatanate blocks

        #self.block1.reshape(-1)

        self.block2 = nn.Sequential(
            nn.Linear(16384, 4608),
            View(-1),
            nn.Linear(4608, 256),
            View(-1),
            nn.Linear(256, 32),
            View(-1))
        # TODO: return torch.cat(self.block1, self.block2, 0): softmax?

    def forward(self, input):
        output = self.block1(input)
        #output = output.view(output.shape[0], -1)
        #output = output.view(-1, 256 * 8 * 8)
        output = self.block2(output)
        return output

'''
# TODO: check this...
    # Initialize model
    modelNew = Simulator()

    # Initialize optimizer
    optimizer = optim.SGD(modelNew.parameters(), lr=0.001, momentum=0.9)

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    # Print optimizer's state_dict
    print("Optimizer's state_dict:")
    for var_name in optimizer.state_dict():
        print(var_name, "\t", optimizer.state_dict()[var_name])
'''