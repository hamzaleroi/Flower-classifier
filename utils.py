#imports
import json
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

# importing the oher files
from create_network import Network
from defaults import *


def load_data(train_dir,valid_dir,test_dir):
    ''' 
        1. Loads data from a given directory after doing the needed pre-processing for the training, validation and testing
        2. Returns a data generator, in small batches.
    '''
    training_transforms = transforms.Compose([
        transforms.RandomRotation(20),
        transforms.RandomResizedCrop(250),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
    ])
    validation_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
    ])
    test_transforms = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),std=(0.229, 0.224, 0.225))
    ])

    # Loading the datasets with ImageFolder
    training_datasets = datasets.ImageFolder(train_dir,transform=training_transforms)
    validation_datasets = datasets.ImageFolder(valid_dir,transform=validation_transforms)
    test_datasets = datasets.ImageFolder(test_dir,transform=test_transforms)

    # Using the image datasets and the transforms, defining the dataloaders
    training_set =  torch.utils.data.DataLoader(training_datasets, batch_size=64, shuffle=True)
    validation_set =  torch.utils.data.DataLoader(validation_datasets, batch_size=32, shuffle=True)
    test_set =  torch.utils.data.DataLoader(test_datasets, batch_size=32, shuffle=True)
    return training_set, validation_set, test_set, training_datasets.class_to_idx

# saving checkpoints to disk
def save_checkpoint(epochs,model,optimizer,hidden_layers,output_size=102,learning_rate=0.001,filename='checkpoint.pth'):
    ''' 
       Saves the model to a file 
    '''
    checkpoint = {
              'output_size': output_size,
              'epochs': epochs,
              'arch':model.architecture,
              'optimizer_dict':optimizer.state_dict(),
              'hidden_layers': hidden_layers,
              'state_dict': model.state_dict(),
              'learning_rate': learning_rate,
              'mapping': model.class_to_idx
             }
    torch.save(checkpoint, filename)
    
# loading checkpoints and creating the model again
def load_checkpoint(filepath,device='cpu'):
    ''' 
        1. Loads the model from an external file, and rebuilds it entirely from that file. A file that have been exported using train.py
        2. Returns the model of class "Network"
    '''
    if device in ['cuda' , 'cpu' ]:
        if not torch.cuda.is_available():
            try :
                checkpoint = torch.load(filepath,map_location='cpu')
            except FileNotFoundError:
                print('checkpoint ' + filepath + ' not found')
                return
            except:
                print('error')
                return
        else:
            checkpoint = torch.load(filepath,map_location='cuda:0')
    else:
        print('error!! unknown device')
        return None
    
    try:
        model =  Network(checkpoint['hidden_layers'],checkpoint['output_size'],checkpoint['arch'])
        model.load_state_dict(checkpoint['state_dict'])
        model.set_class_to_idx(checkpoint['mapping'])
        optimizer = optim.Adam(model.classifier.parameters(), lr=checkpoint['learning_rate'])
        optimizer.load_state_dict(checkpoint['optimizer_dict'])
    except:
        print('error importing the model')
        return None
    return model

# process a PIL image 
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    try:
        im_resized = image.resize((224,224))

        mean=np.array([0.485, 0.456, 0.406])
        std=np.array([0.229, 0.224, 0.225])
        array = np.array(im_resized)

        array = (array - array.mean())/array.std()
        array = array.transpose((2,0,1))
        return array
    except :
        print('error processing the image')
        return None

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def load_mappings(cat_to_name_json_file):
    ''' 
        1. Loads the file that contains the mappings between class number and name of the flower
        2. Returns a Dictionnary object
    '''
    try :
        with open(cat_to_name_json_file, 'r') as f:
            cat_to_name = json.load(f)
        return cat_to_name
    except FileNotFoundError :
        print('file '+ cat_to_name_json_file + ' not found!!')
        return None
    except :
        print('unknown error')
        return None