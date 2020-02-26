# Flower-classifier
This a project that I have done for the AI programming with python nanodegree from udacity.  
I used in this project pytorch, and some pre-trained models for image classification.  
The  following are the features that are provided by the project:
# Training the model:
the Training is done using the command line utility ```train.py```, using the command  
## Syntax:
```bash
train.py [-h] [--save_dir SAVE_DIR] [--learning_rate LEARNING_RATE]
                [--arch ARCH] [--hidden_units HIDDEN_UNITS] [--gpu [GPU]]
                [--epochs EPOCHS]
                [data_directory]
```
## How-to train:
1. first create a folder that contains 3 subfolders: train, test, and valid
2. each of the 3 subfolders, should contain other subfolders indexed according to the matching name-class number in the file cat_to_name.json  
      * example:
      in the json file there is {"21": "fire lily"}, which means that all flowers that are "fire lily" should be put in a folder named 21; and so on.
3. if the folders is named "flowers" you dont need to specify the image folder because thats the default setting. Otherwise, you need to specifiy the path to it.
4. The default architecture is VGG16, but you can still specify one of the following architectures:
      * ALEXNET  
      * VGG11   
      * VGG11_BN   
      * VGG13   
      * VGG13_BN   
      * VGG16   
      * VGG16_BN   
      * VGG19   
      * VGG19_BN   
      * RESNET18   
      * RESNET34   
      * RESNET50   
      * RESNET101   
      * RESNET152   
      * DENSENET121   
      * DENSENET169   
      * DENSENET161   
      * DENSENET201   
      * INCEPTION_V3     
5. Specify a learning rate if another value other than "0.001" is needed.
6. Choose the number of hidden units for the input of the last layer. Here the default value is 512.
7. Choose the number of epochs. The default value is 4.
8. specify the device on which the training will be executed. but specifying the ```--gpu```, the code will be executed on cuda. If the option is not specified it will by default execute on the cpu.
9. specify the directory in which the checkpoint for the model will be saved. By default it will be saved in the same folder as ```train.py``` file . This checkpoint is an image of the model it will allow you to use it to predict latter.

That's all you have to do to be able to train. Some useful information will be displayed on the screen, like the accuracy of the model.

# Prediction:
To predict the class of a flower, use the ```predict.py``` command line utility, like so:
```bash
predict.py [-h] [--top_k TOP_K] [--category_names CATEGORY_NAMES]
                  [--gpu [GPU]]
                  image [checkpoint]
```
### How-to predict:
1. Ensure you have a valid checkpoint created by the ```train.py``` command line utility. The program will assume having a checkpoint in the same directory named ```checkpoint.pth```, if nothing has been specified. To choose a different checkpoint, enter the folder where is the ```checkpoint.pth``` file located. 
2. You can specify some additional options like:
    * --top_k number (default 5): for displaying tok 5 predictions. 
    * --category_names /path/to/file (default ./cat_to_name.json).
    * --gpu (default cpu selected): to excute the rediction on the gpu.
  
