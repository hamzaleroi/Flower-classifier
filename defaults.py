import torchvision 




# available architectures
models = {
    'alexnet'  :  torchvision.models.alexnet,
    'vgg11' :   torchvision.models.vgg11,
    'vgg11_bn' :   torchvision.models.vgg11_bn,
    'vgg13'  :  torchvision.models.vgg13,
    'vgg13_bn' :   torchvision.models.vgg13_bn,
    'vgg16' :   torchvision.models.vgg16,
    'vgg16_bn'  :  torchvision.models.vgg16_bn,
    'vgg19'  :  torchvision.models.vgg19,
    'vgg19_bn'   : torchvision.models.vgg19_bn,
    'resnet18'  :  torchvision.models.resnet18,
    'resnet34'  :  torchvision.models.resnet34,
    'resnet50'  :  torchvision.models.resnet50,
    'resnet101' :   torchvision.models.resnet101,
    'resnet152' :   torchvision.models.resnet152,
    'densenet121'  :  torchvision.models.densenet121,
    'densenet169'  :  torchvision.models.densenet169,
    'densenet161'  :  torchvision.models.densenet161,
    'densenet201'  :  torchvision.models.densenet201,
    'inception_v3'  :  torchvision.models.inception_v3,
}

