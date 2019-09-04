#imports
import argparse
import os
from collections import OrderedDict
import torch
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import json


#define user options here
#initialize parser
#del parser
parser = argparse.ArgumentParser()

option_dict = {
    'data_dir': 'Data directory (preordered as in Udacity\'s source)',
    '--save_dir': 'Location for model checkpoints',
    '--learning_rate' : 'def. 0.003',
    '--hidden_units' : 'def. 521',
    '--epochs' : 'def. 20',
    '--gpu' : 'Enable GPU',
    '--mapping_file' : '.json mapping ID:Name', #for prediction
    '--arch' : 'pretrained architecture, def. \'vgg13\''
}

parser.add_argument("data_dir",
help=option_dict['data_dir'],
type = str)

parser.add_argument('--learning_rate',
help=option_dict['--learning_rate'], type = float)

parser.add_argument('--hidden_units',
help=option_dict['--hidden_units'], type = float)


parser.add_argument("--arch",
help=option_dict['--arch'], type =str)

parser.add_argument("--gpu",
help=option_dict['--gpu'], action ='store_true')

parser.add_argument("--epochs",
help=option_dict['--epochs'],
type = int)

# parser.add_argument("--mapping_file",
# help=option_dict['--mapping_file'], type =str)
args = parser.parse_args()





#pretrained models allowed
# TODO set true
global arch_dict
arch_dict = {
    'resnet18': models.resnet18(pretrained=False),
    'alexnet' : models.alexnet(pretrained=False),
    'vgg16' : models.vgg16(),
    'squeezenet'  : models.squeezenet1_0(pretrained=False),
    'densenet' : models.densenet161(pretrained=False),
    'inception' : models.inception_v3(pretrained=False)
}


#TODO IF
device = 'cpu'
if args.gpu:
    if torch.cuda.is_available():
        device = 'cuda:0'
        print('Cuda found and enabled.')
    else:
        user_chosen_device = 'cpu'
        print('Cuda not found - default to CPU.')
else:
    device = \
    torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Device: ' + str(device))

args = parser.parse_args()

data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

print(data_dir)

if args.hidden_units:
    hidden_units =args.hidden_units #--hidden_units
else:
    hidden_units =512 #--hidden_units

if args.epoch:
    n_epochs =args.epoch
else:
    n_epochs =20
batch_size = 32

#to predictiondir
    #--category_names
#TODO Set -- batch_size

#--learning_rate
if args.learning_rate:
    learning_rate=args.learning_rate
else:
    learning_rate=0.003

#--arch
if args.arch:
    arch =args.arch
else:
    arch = 'vgg13'


data_transforms = {
    'train' : transforms.Compose([transforms.RandomRotation(30),
                                 transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])]),
    'test' : transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])]),

    'valid' :transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])
}

# Load the datasets with ImageFolder
image_datasets = {'train' : datasets.ImageFolder(train_dir, data_transforms['train']),
               'test': datasets.ImageFolder(test_dir, data_transforms['test']),
               'valid': datasets.ImageFolder(valid_dir, data_transforms['valid'])}

# Using the image datasets and the trainforms, define the dataloaders
dataloaders = {'train' : torch.utils.data.DataLoader(image_datasets['train'],
                                                     batch_size = 32,
                                                     shuffle=True),
               'test' : torch.utils.data.DataLoader(image_datasets['test'],
                                                     batch_size = 32,
                                                     shuffle=True),
               'valid': torch.utils.data.DataLoader(image_datasets['valid'],
                                                     batch_size = 32,
                                                     shuffle=True)}

#get model
model = arch_dict[arch]
# Freeze model params
for param in model.parameters():
    param.requires_grad = False

#model
#define default initialized model.
