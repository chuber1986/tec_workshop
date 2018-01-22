import sys

import torch
import torch.nn.functional as f
from torch.autograd import Variable
from torchvision import datasets, transforms

from tec.dnn.architecture import ZOO
from tec.utils.plotting import TrainingPlot
from tec.utils.torch_utils import load_checkpoint, save_checkpoint


def train(model, data_loader, optimizer, epoch):
    """
    Trains a given NN on a given dataset.

    :param model: a Pytorch defined model of a NN
    :param data_loader: iterator through the training dataset
    :param optimizer: Pytorch implementation of an gradient descent algorithm
    :param epoch: max iterations through the dataset
    :return: train accuracy and loss for each iteration
    """
    model.train()
    train_loss = []
    accuracy = []

    # TODO Iterate over the batches losses and accuracies and and perform backpropagation
    for batch_idx, (inputs, labels) in enumerate(data_loader):
        # Prepare input data
        ...

        # Reset the optimizer and perform inference
        ...

        # Compute accuracy
        ...

        # Compute loss and perform backpropagation
        ...

        # Print current training state
        if batch_idx % 10 == 0:
            print('Train Epoch: {} ({:.0f}%)\tLoss: {:.6f}'.format(
                epoch, 100. * batch_idx / len(data_loader), loss.data[0]))

    return accuracy, train_loss


def test(model, data_loader):
    """
    Tests a given NN on a given dataset.

    :param model: a Pytorch defined model of a NN
    :param data_loader: iterator through the validation dataset
    :return: test accuracy and loss
    """
    model.eval()
    test_loss = 0
    correct = 0

    # TODO Iterate over the batches and sum up the the losses and the amount of correct predictions
    for inputs, labels in data_loader:
        # Prepare input data
        ...

        # Perform inference
        ...

        # Compute loss and batch accuracy
        ...

    # Print test accuracy
    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss.data[0], correct, len(data_loader.dataset),
        100. * correct / len(data_loader.dataset)))

    return correct / len(data_loader.dataset), test_loss.data[0]


def main():
    architecture = 0
    data_path = '../../dataset'
    model_path = '../../model'
    resume = False

    max_epochs = 50
    batch_size = 1000
    learning_rate = 1e-2

    ##########################################################################
    #  Load dataset                                                          #
    ##########################################################################

    # TODO Prepare dataloader
    ...

    ##########################################################################
    #  Load model                                                            #
    ##########################################################################

    model = None
    optimizer = None
    epoch = 1
    # TODO Load model, specify an optimizer or load an existing checkpoint
    ...

    print(model)

    ##########################################################################
    #  Main trian loop                                                       #
    ##########################################################################

    # TODO Perform main loop, plot training process, save the model
    ...

    print('Finished execution!')


if __name__ == '__main__':
    main()
