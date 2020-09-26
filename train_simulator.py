import torch
import torchvision
import time
import code_idan
import Simulator
import torch.nn as nn
from torch import optim
import datetime
from datetime import date
from datetime import datetime
import sys
import os


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
BASE_DIR = 'C:/Users/Idan/PycharmProjects/data/'
# BASE_DIR = '/content/drive/My Drive/DeepLearningIdan/project/'
LOG_DIR = 'C:/Users/Idan/PycharmProjects/data/logs/'

BATCH_SIZE = 3


def trainS(training_loader):
    netS = Simulator.Simulator()
    netS.train()
    loss_criterion = nn.MSELoss()
    optimizerS = optim.Adam(netS.parameters(), lr= 1e-6, betas=(0.5, 0.999))
    # ? optimizerS = optim.SGD(netS.parameters(), lr= 1e-6, momentum = 0.9, weight_decay= 0.01)

    S_loss = 0
    min_loss = 1000
    min_loss_epoch = -1

    epochs = 3

    today = date.today()
    now = datetime.now()
    time_stamp = today.strftime("%d_%m_%y")
    time_stamp2 = now.strftime("%H_%M")
    log_file_name = f"{time_stamp}_{time_stamp2}.txt"

    # done: load for the good weights of the right model (if is_load_model = 1 then load a previouus model and continue from there)
    # TODO: save the model every epoch in case that the loss decreases and then increases
    # TODO: add validation set to run on, and save as validation loss

    # didn't work: skimage, torchvision (missing DLLs). maybe find a conda installation of torchvision,
    #  and not a 'pip'. if true, uninstall the pip installation form the virtual environment.

    # TODO: calculate ""epochs"" (in the next row) from the parameters

    # somehow initialize S parameters differently?
    #is_load_prev_model = input("Type 1 for previous model, 0 otherwise")
    #if is_load_prev_model == 1:
    #    modelS_path= '0' # TODO: enter by the user?
    #    netS.load_state_dict(torch.load(modelS_path))
    #else:
    log_file = open(f"C:/Users/Idan/PycharmProjects/data/logs/{log_file_name}", "w")

    for epoch in range(0, epochs):
        train_loss = 0
        pics = 0
        for batch_idx, data in enumerate(training_loader, 0):
            log_file.write('.\nEpoch: {}, Step: {}'.format(epoch + 1, batch_idx))
            # print(data['metadata_tensor'])
            # print(data['image_tensor'])
            print(data['metadata_tensor'])
            optimizerS.zero_grad()
            simulated_spec = netS(data['image_tensor'])
            S_loss = loss_criterion(data['metadata_tensor'], simulated_spec)
            S_loss.backward()
            optimizerS.step()
            train_loss += S_loss.item()
            if S_loss.item() < min_loss:
                min_loss = S_loss.item()
                min_loss_epoch = epoch
            print(f"batch #{batch_idx+1}, epoch {epoch+1}, Simulator loss is:" + str(S_loss.item()))
            log_file.write(", loss: %.5f" % (train_loss))
            log_file.write("\n")
        train_loss = train_loss / (batch_idx+1)
        print("Epoch loss was: " + str(train_loss))
        print("------------------------------------------------")
# print("Epoch with lowest loss was: " + str(min_loss_epoch) + " and its loss was: " + str(min_loss))

    log_file.close()
    #should i find minimum loss and its epoch?
    # SAVE Simulator Model in case the loss decreases
    model_S_path = 'C:/Users/Idan/PycharmProjects/data/models'
    model_S_name = '{}{}{:d}{}'.format(model_S_path, "net_", epoch + 1, ".pth")

    # torch.save(netS.state_dict(), model_S_name + '_S.pt')
    torch.save(netS.state_dict(), model_S_name)
    return (S_loss, simulated_spec) # need to return the lowest loss!


if __name__ == "__main__":
    dataset_root_dir = 'C:/Users/Idan/PycharmProjects/data/'
    training_loader, test_loader = code_idan.generate_training_set(dataset_root_dir)
    (S_loss, returned_spec) = trainS(training_loader)
    print(str(returned_spec))
    print(str(S_loss.item()))
    sys.exit()
