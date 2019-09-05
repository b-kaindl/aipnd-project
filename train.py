#imports
import argparse
import os
from collections import OrderedDict
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import json

#functions -migrate eventually
def train_model(model,optimizer,criterion,dataloader,n_classes, n_hidden, drop_p,
learn_rate,cat_to_name, n_epochs=1,
device='cpu',save_dir =''):
    print('Start Training')
    running_loss = 0
    steps = 0
    print_every = 20
    train_size = len(dataloader)
    model.train()
    for epoch in range(n_epochs):
        for images, flowers in dataloader:
            images, flowers = images.to(device), flowers.to(device)

            optimizer.zero_grad()

            logps = model.forward(images)
            loss = criterion(logps, flowers)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            steps += 1

            if steps % print_every == 0:
                    print(f"Epoch {epoch+1}/{n_epochs}.."
                      f"Train Loss: {running_loss/steps:.3f}..")


    model.class_to_idx =cat_to_name
    checkpoint = {
    'input_size' : model.classifier[0].in_features,
    'output_size' : n_classes,
    'hidden_size' : n_hidden,
    'state_dict' : model.classifier.state_dict(),
    'epochs': n_epochs,
    'learn_rate': learn_rate,
    'optimizer' : optimizer,
    'optim_state': optimizer.state_dict(),
    'dropout_prob': drop_p,
    'criterion' : loss,
    'class_to_idx': model.class_to_idx
    }



    torch.save(checkpoint, save_dir + 'checkpoint.pth')


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
    '--arch' : 'pretrained architecture, def. \'vgg16\''
}

parser.add_argument("data_dir",
help=option_dict['data_dir'],
type = str)

parser.add_argument("--save_dir",
help=option_dict['--save_dir'],
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

parser.add_argument("--mapping_file",
help=option_dict['--mapping_file'], type =str)
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

if args.mapping_file:
    with open(args.mapping_file, 'r') as f:
        cat_to_name = args.mapping_file
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
if args.hidden_units:
    hidden_units =args.hidden_units #--hidden_units
else:
    hidden_units =512 #--hidden_units

if args.epochs:
    n_epochs =args.epochs
else:
    n_epochs =5

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
    arch = 'vgg16'

if args.save_dir:
    save_dir = args.save_dir
else:
    save_dir=''


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
                                                     batch_size = batch_size,
                                                     shuffle=True),
               'test' : torch.utils.data.DataLoader(image_datasets['test'],
                                                     batch_size = batch_size,
                                                     shuffle=True),
               'valid': torch.utils.data.DataLoader(image_datasets['valid'],
                                                     batch_size = batch_size,
                                                     shuffle=True)}

#get model
model = arch_dict[arch]

# Freeze model params
for param in model.parameters():
    param.requires_grad = False

drop_p =.2
n_inputs = model.classifier[0].in_features
n_hidden =hidden_units
n_classes =102

flower_classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(n_inputs,n_hidden)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(n_hidden,n_classes)),
    ('dropout', nn.Dropout(drop_p)),
    ('logsoftmax', nn.LogSoftmax(dim=1))
]))

model.classifier = flower_classifier

#loss function, opimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)

model.to(device)

train_model(model,optimizer,criterion,dataloaders['train'],
n_classes = n_classes, n_hidden=n_hidden, drop_p=drop_p,
learn_rate=learning_rate,cat_to_name = cat_to_name, n_epochs=n_epochs,
device = device ,save_dir = save_dir)
