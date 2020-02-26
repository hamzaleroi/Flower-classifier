#imports
import torch 
import argparse
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
import create_network
from defaults import *
from  utils import *
import training_routines 




def get_input_args():
    ''' 
        1. Read command line arguments and convert them into the apropriate data type. 
        2. Returns a data structure containing everything that have been read, or the default values 
        for the paramater that haven't been explicitly specified.
    '''
    parser = argparse.ArgumentParser()
    
    parser.add_argument('image',  type = str, help = 'image path')
    
    parser.add_argument('checkpoint', nargs='?', type = str, default = 'checkpoint.pth', help = 'checkpoint path to use')
    
    parser.add_argument('--top_k', type = int, default = 3, help = 'Top K predictions') 

    parser.add_argument('--category_names', type = str, default = 'cat_to_name.json', help = 'JSON file that contains categories mapping') 
    
    parser.add_argument('--gpu', const=True, nargs='?', default = False, help = 'Use GPU or not')


    in_args = parser.parse_args()

    return in_args

args = get_input_args()


# default values
path_to_image=args.image
checkpoint_path = args.checkpoint
cat_to_name_json_file = args.category_names
if args.gpu :
    device='cuda'
else:
    device='cpu'
topk=args.top_k

#prediction function
def predict(image_path, model, topk=5, device='cpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    if device in ['cuda' , 'cpu' ]:
        if torch.cuda.is_available():
            model.to(device)
        else:
            if device == 'cuda' :
                print('error!!! cuda not available, switching to  CPU...  ')
                device = 'cpu'
            model.to('cpu')
    else:
        print('error!! unknown device')
        return None
    model.eval()
    try:
        im = Image.open(image_path)
    except FileNotFoundError:
        print('image file not found!!')
        return None
    except:
        print('unknown error')
        return None
    array = np.array([process_image(im)])
    input_image = torch.tensor(array)
    input_image = input_image.float()
    if torch.cuda.is_available():
        model.to(device)
        output = model.forward(input_image.to(device))
        result = torch.exp(output).to('cpu')
        if topk <= result.size()[1]:
            probs = result.topk(topk)[0][0].tolist()
            classes = result.topk(topk)[1].numpy()
            return probs,[c[0] for c in model.class_to_idx.items() if c[1] in classes ]
        else:
            return result.sort()[0][0]
    else:
        model.to('cpu')
        output = model.forward(input_image.to(device))
        result = torch.exp(output).to('cpu')
        if topk <= result.size()[1]:
            probs = result.topk(topk)[0][0].tolist()
            classes = result.topk(topk)[1].numpy()
            return probs,[c[0] for c in model.class_to_idx.items() if c[1] in classes ]
        else:
            return result.sort()[0][0]
        

def init():
    ''' Runs the prediction, depending on the command line arguments been read
    '''
    cat_to_name = load_mappings(cat_to_name_json_file)
    if cat_to_name == None:
        exit()
    model = load_checkpoint(checkpoint_path,device=device)
    if model == None:
        exit()
    result = predict(path_to_image,model,topk,device=device)
    
    
    if result != None:
        probs, classes = result
    else:
        exit()
    print('\n[results for image: ' + path_to_image + ']:\n')
    names = [cat_to_name[c] for c in classes]
    for i in range(len(probs)):
        print(' class ({}) : {} -----> with probability ({:.2f}%)  '.format(classes[i],names[i],probs[i]*100))

init()
