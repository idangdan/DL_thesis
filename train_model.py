import os, sys
import time
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader
import re
import glob
from torch import nn
from torch import autograd
from torch import optim
from torch.autograd import Variable
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from matplotlib import image as img
from Simulator import Simulator as sim


def trainD_batch(batch_real_data):
    this_batch_size = batch_real_data.shape[0]
    netD.zero_grad()  # what does zero_grad mean??

    # real
    batch_real_data = batch_real_data.to(device)
    D_real = netD(batch_real_data)

    # fake
    noise = torch.randn(this_batch_size, 128).to(device)
    with torch.no_grad():
        noisev = autograd.Variable(noise)  # totally freeze netG
        fake = autograd.Variable(netG(noisev).data)
    D_fake = netD(fake)
    gradient_penalty = calc_gradient_penalty(netD, batch_real_data.data, fake.data)
    loss = D_fake.mean() - D_real.mean() + gradient_penalty
    loss.backward()
    optimizerD.step()
    return loss


def trainG_batch():
    netG.zero_grad()
    noise = torch.randn(G_batch_size, 50).to(device)
    noisev = autograd.Variable(noise)
    fake = netG(noisev + input_spectra)
    D_output = netD(fake)
    S_output = netS(fake)

    loss = -(S_output.mean() + S_D_factor * D_output.mean())  # Is the minus sign needed?
    loss.backward()
    optimizerG.step()
    return loss


DIM = 64  # 128 overfits substantially; you're probably better off with 64
LAMBDA = 10  # Gradient penalty lambda hyperparameter
CRITIC_ITERS = 5  # How many critic iterations per generator iteration
BATCH_SIZE = 64  # Batch size
ITERS = 20000  # How many generator iterations to train for ## 20000???
CHANNELS = 1
IM_SIZE = 64
OUTPUT_DIM = CHANNELS * IM_SIZE * IM_SIZE  # Number of pixels in data pictures
S_batch_size = 128
D_batch_size = 16
G_batch_size = 16
spectrum_size = 32  # frequency samples
S_D_factor = 0.0001  # feedback factor for losses, as G_loss= S_loss+ S_D_factor* D_loss


device = torch.device("cuda")
print('a')
netS = sim().block1.to(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
netD = sim().block1.to(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
netG = sim().block1.to(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

SPECTRUM_SIZE = 32


BASE_DIR = 'C:/Users/Idan/PycharmProjects/data/'
# BASE_DIR = '/content/drive/My Drive/DeepLearningIdan/project/'
LOG_DIR = 'C:/Users/Idan/PycharmProjects/data/logs/'

optimizerD = optim.Adam(netD.parameters(), lr=1e-4, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=1e-4, betas=(0.5, 0.999))

# DONE: load pre-trained Simulator weights
modelS_path = '/data/models/BEST_PATH'
netS.load_state_dict(torch.load(modelS_path))

n_epochs = (ITERS * CRITIC_ITERS) // len(train_loader)
D_losses = []
G_losses = []


# TODO for future: create a combined net with G+D (G trainable false)
# start_time = time.time()
G_trainable = True # can also be False


for epoch in range(n_epochs):
    # train epoch
    for batch_idx, (batch_real_data, _) in enumerate(train_loader):
        # train D
        # reset requires_grad
        for p in netD.parameters():
            p.requires_grad = True  # they are set to False below in netG update
        D_loss = trainD_batch(batch_real_data)
        D_losses.append(D_loss)

        # train G every CRITIC_ITERS batches
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
        if (G_trainable):
            if (batch_idx % CRITIC_ITERS == 0) & (batch_idx > 0):
                G_loss = trainG_batch()
                G_losses.append(G_loss)
                if (batch_idx % 100 == 0):
                    print('batch: ' + str(batch_idx) + ' D loss: ' + str(D_loss.item()) + ' G loss: ' + str(G_loss.item()))


        # DONE: create log with loss and epoch iteration
        log_file = open(LOG_DIR, "simulator_model")
        log_file.write('.\nEpoch: {}, Step: {}'.format(epoch+1, i))
        log_file.write(", loss: %.5f" % (running_loss / (count + 1)))
        log_file.write("\n")
        log_file.close()

    # SAVE MODELS
    model_path = BASE_DIR + '/models'
    new_name = '{}{}{:d}{}'.format(model_path, "net_", epoch + 1, ".pth")
    torch.save(netS.state_dict(), new_name + '_S.pt')  # what state_dict is????
    torch.save(netG.state_dict(), new_name + '_G.pt')
    torch.save(netD.state_dict(), new_name + '_D.pt')

# loss visualization
D_losses_plot = np.array(D_losses)[np.arange(0, len(D_losses), CRITIC_ITERS).astype(np.int)]
plt.figure()
plt.plot(G_losses, label='G')
plt.plot(D_losses_plot, label='D')
plt.legend()
plt.xlabel('Generator iteration')
plt.ylabel('Loss')
# plt.title(MODE)
plt.savefig(BASE_DIR + '/models/figures/' + '.png')
plt.show()