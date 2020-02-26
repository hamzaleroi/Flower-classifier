#imports
import torch 
from torch import nn 
from torch import optim 
import numpy as np
import math
from PIL import Image
import torchvision
from torchvision import datasets, transforms
from matplotlib import colors, cm, pyplot as plt
from collections import OrderedDict

from defaults import models

# importing a model of a given archirtecture
def initialize_model(arch='vgg16'):
    ''' 
        1. Loads a predefined model af a given architecture, by default vgg16
        2. Returns the model with all the weights frozen
    '''
    if arch != None and arch.lower() in models:
        model  = models[arch](pretrained=True)
    else :
        print('architecture {} is not valid'.format(arch))
        return None
    for param in model.parameters():
        param.requires_grad = False
    return model



# Stacking up the layers

def stacking_layers(input_size,hidden_layers,output_size):
    ''' 
        1. Creates a stack of linear layers with dropout and ReLU for first layers and LogSoftmax for
        the rest
        2. Returns a python list of sets (name of the layer, its function)
    '''
    if len(hidden_layers) == 0:
        return [('fc1', nn.Linear(input_size, output_size)), 
               ('output', nn.LogSoftmax(dim=1))]
    layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
    layers  = [('fc1', nn.Linear(input_size, hidden_layers[0])), 
               ('relu', nn.ReLU(inplace=True)),
               ('drop', nn.Dropout())
              ]      
    index=-1
    for index, (h1, h2) in enumerate(layer_sizes):
        layers.append((('fc'+str(index+2), nn.Linear(h1, h2))))
        layers.append(('relu'+str(index+2), nn.ReLU(inplace=True)))
        layers.append(('drop'+str(index+2), nn.Dropout(p=.2)))

    layers.append((('fc'+str(index+3), nn.Linear(hidden_layers[-1], output_size))))
    layers.append(('output', nn.LogSoftmax(dim=1))) 
    return layers


# Creating the classifier, and adding it to the predefined model:

def create_model(hidden_layers,output_size,arch='vgg16'):
    ''' 
        1. Creats the model with the desired customization 
        2. Returns the model.
    '''
    model = initialize_model(arch)
    if model == None:
        return None
    if hasattr(model, 'classifier'):
        input_size = -1
        # if the classifier is just a layer
        if  isinstance(model.classifier, torch.nn.modules.linear.Linear):
            input_size = model.classifier.in_features
        else:
        # if the classifier contains multiple layers
            for layer in model.classifier:
                if  isinstance(layer, torch.nn.modules.linear.Linear):
                    input_size = layer.in_features
                    break
        # unknown architecture
        if input_size == -1:
            return None
        classifier = nn.Sequential(OrderedDict(stacking_layers(input_size,hidden_layers,output_size)))
        model.classifier = classifier
    elif hasattr(model, 'fc'):
        input_size = -1
        # if the classifier is just a layer
        if  isinstance(model.fc, torch.nn.modules.linear.Linear):
            input_size = model.fc.in_features
        else:
        # if the classifier contains multiple layers
            for layer in model.fc:
                if  isinstance(layer, torch.nn.modules.linear.Linear):
                    input_size = layer.in_features
                    break
        # unknown architecture
        if input_size == -1:
            return None
        classifier = nn.Sequential(OrderedDict(stacking_layers(input_size,hidden_layers,output_size)))
        model.fc = classifier
    else:
        return None
    return model


# the model class 
class Network(nn.Module):
    def __init__ (self,hidden_layers,output_size,arch='vgg16'):
        super().__init__()
        self.class_to_idx=None
        self.architecture  = arch
        self.model = create_model(hidden_layers,output_size,arch)
        self.classifier = None
        if hasattr(self.model, 'classifier'):
            self.classifier = self.model.classifier
        elif hasattr(self.model, 'fc'):
            self.classifier = self.model.fc
        else:
            return None
    def forward(self,x):
        return self.model.forward(x)
    def load_state_dict(self,state_dict):
        self.model.load_state_dict(state_dict)
    def state_dict(self):
        return self.model.state_dict()
    def set_class_to_idx(self, class_to_idx):
        ''' 
        1. Sets the mappings name-class_number locally in the class.
        '''
        self.class_to_idx = class_to_idx        

