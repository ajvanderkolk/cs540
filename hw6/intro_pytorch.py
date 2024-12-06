import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms


def get_data_loader(training = True):
    """
    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    # data is 28x28 images of fashion items

    custom_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    data_set = datasets.FashionMNIST('./data',
                                        train=training, 
                                        download=True,
                                        transform=custom_transform
                                    )
    loader = torch.utils.data.DataLoader(data_set, batch_size = 64)

    return loader



def build_model():
    """
    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    # flattened 28x28 pixel images -> 784 input neurons
    # 128 hidden neurons -> 64 hidden neurons -> 10 output neurons
    # ReLU activation function for hidden layers
    # using Sequential model
    
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model



def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """


if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''
    criterion = nn.CrossEntropyLoss()
    ## test get_data_loader
    # train_loader = get_data_loader()
    # print(type(train_loader))
    # print(train_loader.dataset)

    ## test build_model
    model = build_model()
    print(model)
