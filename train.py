#imports
import argparse
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

# importing other files
from create_network import Network
from defaults import *
from utils import *
from training_routines import *


# getting input arguments
def get_input_args():
    ''' 
        1. Read command line arguments and convert them into the apropriate data type. 
        2. Returns a data structure containing everything that have been read, or the default values 
        for the paramater that haven't been explicitly specified.
    '''
    parser = argparse.ArgumentParser()
    
    parser.add_argument('data_directory', nargs='*', type = str, default=['flowers'], help = 'training, validation, and testing data folder')
    
    parser.add_argument('--save_dir', type = str, default = '.', help = 'checkpoint path to use')
    
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'The learning rate of the model') 

    parser.add_argument('--arch', type = str, default = 'vgg16', help = 'Model Architecture') 

    parser.add_argument('--hidden_units',  type=int, default = 512, help = 'Number of hidden units') 
    
    parser.add_argument('--gpu', const=True, nargs='?', default = False, help = 'Use GPU or not') 
    
    parser.add_argument('--epochs', type = int, default = 4, help = 'Number of training epochs') 
    
     


    in_args = parser.parse_args()

    return in_args

args = get_input_args()

#training parameters
data_dir = args.data_directory[0].rstrip('/')
save_dir = args.save_dir.rstrip('/')+ '/checkpoint.pth'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
output_size  = 102
hidden_layers=[args.hidden_units]
epochs = args.epochs
print_every = 40
learning_rate=args.learning_rate
architecture = args.arch

if args.gpu :
    device='cuda'
else:
    device='cpu'


def init():
    ''' Runs the training, depending on the command line arguments read
    '''
    print('using architecture "{}" with {} units'.format(architecture,hidden_layers[0] ))
    model  = Network(hidden_layers,output_size,architecture)
    
    if model.model == None:
        print('model failed to be created !!')
        return
    print('loading data from directory "{}" ...'.format(data_dir))
    result = load_data(train_dir,valid_dir,test_dir)
    if result != None:
        training_set, validation_set, test_set,class_to_idx = result
    else:
        print('error !!! data not loaded')
        return 
    model.set_class_to_idx(class_to_idx)

    # defining parameters for the model:
    print('setting parameters:\n      creterion: "NNLoss", \n      optimizer: "Adam" with learning rate "{}"'.format(learning_rate))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # training 
    print('training....')
    save_checkpoint(epochs,model,optimizer,hidden_layers,output_size,learning_rate=learning_rate,filename='checkpoint.pth')
    model.train()
    train(epochs,model,training_set,validation_set,print_every,optimizer,criterion,device)

    # evaluation
    print('evaluating...')
    model.eval()
    test_loss, accuracy=validation(model,test_set,criterion,device)
    print("Test Loss: {:.3f}.. ".format(test_loss),"Test Accuracy: {:.3f}%".format(accuracy*100))

    print('saving to directory "{}" ...'.format(save_dir))
    save_checkpoint(epochs,model,optimizer,hidden_layers,output_size,learning_rate=learning_rate,filename='checkpoint.pth')

init()





