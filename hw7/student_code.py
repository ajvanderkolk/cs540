# python imports
import os
from tqdm import tqdm

# torch imports
import torch
import torch.nn as nn
import torch.optim as optim

# helper functions for computer vision
import torchvision
import torchvision.transforms as transforms


class LeNet(nn.Module):
    def __init__(self, input_shape=(32, 32), num_classes=100):
        super(LeNet, self).__init__()

        # 1. One convolution layer with 6 output channels, kernel size=5, stride=1, 
        #   followed by a ReLU activation and 
        #   2D max pool operation, kernel size=2, stride=2
        # 2. One conv layer with 16 output channels, kernel size = 5, stride = 1, 
        #   followed by a ReLU activation
        #   and a 2D max pool operation (kernel size = 2 and stride = 2).
        # 3. A flatten layer to convert the 3D tensor to a 1D tensor.
        # 4. A linear layer with output dimension = 256, followed by a ReLU activation.
        # 5. A linear layer with output dimension = 128, followed by a ReLU activation.
        # 6. A linear layer with output dimension = number of classes (in our case, 100).

        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv_2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        
        # self.relu_2 = nn.ReLU()
        # self.pool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.flatten_layer = nn.Flatten()
        # No parameters are needed; reshape tesnor in forward

        self.fc1 = nn.Linear((16 * 5 * 5), 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        shape_dict = {}
        
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.pool(x)
        shape_dict[1] = list(x.shape) # Store shape after first block

        x = self.conv_2(x)
        x = self.relu(x)
        x = self.pool(x)
        shape_dict[2] = list(x.shape)

        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        shape_dict[3] = list(x.shape)  # Store shape after flattening
        
        x = self.fc1(x)
        x = self.relu(x)
        shape_dict[4] = list(x.shape)  # Store shape after first linear layer
        
        x = self.fc2(x)
        x = self.relu(x)
        shape_dict[5] = list(x.shape)  # Store shape after second linear layer
        
        x = self.fc3(x)
        shape_dict[6] = list(x.shape)  # Store final output shape

        out = x

        return out, shape_dict


def count_model_params(test_output=False):
    '''
    return the number of trainable parameters of LeNet.
    '''
    model = LeNet()
    model_params = 0.0

    if test_output:
        for name, param in model.named_parameters():
            print(f"Name: {name}")
            print(f"Parameter: {param}")
            print(f"Shape: {param.shape}\n")
        model_params = -1.0
    else:
        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        

    return model_params / 1e6

# in convolutional layers, weights are shared across the spatial dimensions (using kernels (filters)) over each input
# weights are reused as it moves over the input - significantly reducing the number of parameters (and computation required)
# e.g. For a simple comparison, imagine a fully connected layer connecting 1,024 neurons to another 1,024 neurons. This requires over a million parameters (1,024 x 1,024). 
# In contrast, a convolutional layer with 16 filters, each 3x3, would require only 144 parameters (16 x 3 x 3), dramatically fewer than the fully connected layer.


def train_model(model, train_loader, optimizer, criterion, epoch):
    """
    model (torch.nn.module): The model created to train
    train_loader (pytorch data loader): Training data loader
    optimizer (optimizer.*): A instance of some sort of optimizer, usually SGD
    criterion (nn.CrossEntropyLoss) : Loss function used to train the network
    epoch (int): Current epoch number
    """
    model.train()
    train_loss = 0.0
    for input, target in tqdm(train_loader, total=len(train_loader)):
        # 1) zero the parameter gradients
        optimizer.zero_grad()
        # 2) forward + backward + optimize
        output, _ = model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Update the train_loss variable
        # .item() detaches the node from the computational graph
        train_loss += loss.item()

    train_loss /= len(train_loader)
    print('[Training set] Epoch: {:d}, Average loss: {:.4f}'.format(epoch+1, train_loss))

    return train_loss


def test_model(model, test_loader, epoch):
    model.eval()
    correct = 0
    with torch.no_grad():
        for input, target in test_loader:
            output, _ = model(input)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = correct / len(test_loader.dataset)
    print('[Test set] Epoch: {:d}, Accuracy: {:.2f}%\n'.format(
        epoch+1, 100. * test_acc))

    return test_acc
