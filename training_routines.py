
# Defining a function that does one pass over one batch of images:
#imports
import torch 
from torch import nn 
from torch import optim 
import numpy as np
import math
from PIL import Image
import torchvision
from torchvision import datasets, transforms, models
from matplotlib import colors, cm, pyplot as plt
from collections import OrderedDict


def training_on_batch(model, optimizer, criterion, image,label,device='cuda'):
    ''' 
        1. Trains the model on a small batch of data.
        2. Returns the loss returned by the loss function.
    '''
    if device in ['cuda' , 'cpu' ]:
        if torch.cuda.is_available():
            model.to(device)
        else:
            if device == 'cuda' :
                print('error cuda not available !! switching to  CPU...  ')
                device = 'cpu'
            model.to('cpu')
    else:
        print('error!! unknown device')
        return None
    
    optimizer.zero_grad()
    output = model.forward(image.to(device))
    
    loss = criterion(output, label.to(device))
    loss.backward()
    
    optimizer.step()
    return loss.item()


#  Defining the validation function:

def validation(model, validation_set, criterion, device='cuda'):
    ''' 
        1. Validates the model accuracy on the whole validation data.
        2. Returns the accuracy and the validation loss
    '''
    test_loss = 0
    accuracy = 0
    step=0
    if device in ['cuda' , 'cpu' ]:
        if torch.cuda.is_available():
            model.to(device)
        else:
            if device == 'cuda' :
                print('error cuda not available !! switching to  CPU...  ')
                device = 'cpu'
            model.to('cpu')
    else:
        print('error!! unknown device')
        return None
    with torch.no_grad():
        for images, labels in validation_set:
            model.eval()
            step+=1
            images, labels = images.to(device), labels.to(device)
            output = model.forward(images)
            test_loss += criterion(output, labels).item()
            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss/step, accuracy/step


# Defining the training function :

def train(epochs,model,training_set,validation_set,print_every,optimizer,criterion,device='cuda'):
    ''' 
        1. Trains the model on the data.
        2. Outputs training loss, validation loss and the validation accuracy.
    '''
    step=1
    running_loss=0
    if device in ['cuda' , 'cpu' ]:
        if torch.cuda.is_available():
            model.to(device)
        else:
            if device == 'cuda' :
                print('error cuda not available !! switching to  CPU...  ')
                device = 'cpu'
            model.to('cpu')
    else:
        print('error!! unknown device')
        return None
    for i in range(epochs):
        for index, (image, label) in enumerate(training_set):
            model.train()
            loss=training_on_batch(model.to(device), optimizer, criterion, image.to(device),label.to(device),device)
            step+=1
            running_loss+=loss
            if step % print_every == 0 :
                model.eval()
                test_loss, accuracy=validation(model,validation_set,criterion)
                print("Epoch: {}/{}.. ".format(i+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Test Loss: {:.3f}.. ".format(test_loss),
                      "Test Accuracy: {:.3f}%".format(accuracy*100))
                running_loss=0

