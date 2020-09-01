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

BATCH_SIZE = 32


def trainS(training_loader):
    #geometry batch is a dictionary of a picture and relevant spectrum - how do I get to each of them??

    # need to take batch of images
    # perform the Snet on them and get the simulated spectrum
    # then take the relevant spectrums of the batch images and calculate the loss
    netS = Simulator.Simulator()
    l2criterion = nn.MSELoss()
    optimizerS = optim.Adam(netS.parameters(), lr= 1e-6, betas=(0.5, 0.999))
    S_loss = 0
    min_loss = 1000
    min_loss_epoch = -1

    epochs = 3

    today = date.today()
    now = datetime.now()
    time_stamp = today.strftime("%d_%m_%y")
    time_stamp2 = now.strftime("%H_%M")
    # print(f"the date is  {time_stamp} and the time is  {time_stamp2}")
    log_file_name = f"{time_stamp}_{time_stamp2}.txt"


    # done: to do another for loop for epochs, and save to log.txt the losses, and see how it trains itself
    # done: load for the good weights of the right model (if is_load_model = 1 then load a previouus model and continue from there)
    # done: save the model every epoch in case that the loss decreases and then increases
    # TODO: add validation set to run on, and save as validation loss

    # didn't work: skimage, torchvision (missing DLLs). maybe find a conda installation of torchvision,
    #  and not a 'pip'. if true, uninstall the pip installation form the virtual environment.

    is_load_prev_model = 0 # 0/1

    # TODO: calculate ""epochs"" (in the next row) from the parameters

    if is_load_prev_model == 1:
        modelS_path= '0' # TODO: enter by the user?
        netS.load_state_dict(torch.load(modelS_path))

    #somehow initialize S parameters differently?

    log_file = open(f"C:/Users/Idan/PycharmProjects/data/logs/{log_file_name}", "w")


    for epoch in range(0, epochs):
        S_loss_running = 0;
        print("-------------------------------------")
        for i, data in enumerate(training_loader, 0):
            print(i)
            print("5!")
            log_file.write('.\nEpoch: {}, Step: {}'.format(epoch + 1, i))
            simulated_spec = netS(data['image_tensor'])
            # print(data['image_tensor'])
            # print(data['metadata_tensor'][0])
            S_loss = l2criterion(data['metadata_tensor'], simulated_spec)
            ######################################### print(f"S loss for epoch: {epoch} is: {S_loss}")
            S_loss.backward()
            optimizerS.step()
            S_loss_running = S_loss.item()
            # if S_loss.item() < min_loss:
            #     min_loss = S_loss.item()
            #     min_loss_epoch = epoch

            print(f"pic #{i+1}, epoch {epoch+1}, Simulator loss is:" + str(S_loss.item()))
            log_file.write(", loss: %.5f" % (S_loss_running))
            log_file.write("\n")

        # print("Epoch loss was: " + str(S_loss_running//BATCH_SIZE))
        # print("Epoch with lowest loss was: " + str(min_loss_epoch) + " and its loss was: " + str(min_loss))

        # creating log with loss and epoch iteration

       # log_file.write(", loss: %.5f" % (S_loss_running / (epochs + 1)))  ## why did i divide it in count?...
    log_file.close()
    #should i find minimum loss and its epoch?
    # SAVE Simulator Model in case the loss decreases
    model_S_path = 'C:/Users/Idan/PycharmProjects/data/models'
    model_S_name = '{}{}{:d}{}'.format(model_S_path, "net_", epoch + 1, ".pth")

    # torch.save(netS.state_dict(), model_S_name + '_S.pt')
    torch.save(netS.state_dict(), model_S_name)
    return (S_loss, simulated_spec)


if __name__ == "__main__":
    dataset_root_dir = 'C:/Users/Idan/PycharmProjects/data/'
    training_loader, test_loader = code_idan.generate_training_set(dataset_root_dir)
    (S_loss, returned_spec) = trainS(training_loader)
    print("HELLO ITS ME " + str(S_loss))
    print(str(returned_spec))
    print(str(S_loss.item()))

    sys.exit()
