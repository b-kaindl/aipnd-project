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
import torch.utils.model_zoo as model_zoo


#functions -migrate eventually
def train_model(model,optimizer,criterion,dataloader,n_classes, n_hidden, drop_p,
learn_rate,cat_to_name, n_epochs=1,
device='cpu',save_dir ='', arch = 'vgg16'):
    print('Start Training')
    running_loss = 0
    steps = 0
    print_every = 10
    test_loss = 0
    accuracy = 0
    train_size = len(dataloader['train'])
    test_size = len(dataloader['test'])
    model.train()
    for epoch in range(n_epochs):
        for images, flowers in dataloader['train']:
            steps +=1
            images, flowers = images.to(device), flowers.to(device)

            optimizer.zero_grad()

            logps = model.forward(images)
            loss = criterion(logps, flowers)
            loss.backward()
            optimizer.step()


            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    test_size = len(dataloaders["test"])
                    for images, flowers in dataloaders['test']:

                        images, flowers = images.to(device), flowers.to(device)
                        logps = model.forward(images)
                        batch_loss = criterion(logps, flowers)
                        test_loss += batch_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim =1)
                        is_match = top_class == flowers.view(*top_class.shape)
                        accuracy += torch.mean(is_match.type(torch.cuda.FloatTensor)).item()
                print(f"Epoch {epoch+1}/{n_epochs}.."
                  f"Train Loss: {running_loss/steps:.3f}.."
                  f"Test Loss: {test_loss/train_size:.3f}.."
                  f"Avg. Test accuracy: {accuracy/test_size:.3f}")
                running_loss = 0
                model.train()


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
    'class_to_idx': model.class_to_idx,
    'arch' : arch
    }


    #save trained model

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
global arch_dict
arch_dict = {
    'resnet18': models.resnet18(pretrained=False),
    'alexnet' : models.alexnet(pretrained=False),
    'vgg16' : models.vgg16(),
    'squeezenet'  : models.squeezenet1_0(pretrained=False),
    'densenet' : models.densenet161(pretrained=False),
    'inception' : models.inception_v3(pretrained=False)
}
#URLS for statedicts
#From https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/torchvision_models.py
model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'densenet121': 'http://data.lip6.fr/cadene/pretrainedmodels/densenet121-fbdb23505.pth',
    'densenet169': 'http://data.lip6.fr/cadene/pretrainedmodels/densenet169-f470b90a4.pth',
    'densenet201': 'http://data.lip6.fr/cadene/pretrainedmodels/densenet201-5750cbb1e.pth',
    'densenet161': 'http://data.lip6.fr/cadene/pretrainedmodels/densenet161-347e6b360.pth',
    'inceptionv3': 'https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth',
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
    # 'vgg16_caffe': 'https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg16-00b39a1b.pth',
    # 'vgg19_caffe': 'https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg19-d01eb7cb.pth'
}


print('Preparing Pretrained models..')



#set Device
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
    hidden_units =args.hidden_units
else:
    hidden_units =512

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
model.load_state_dict(model_zoo.load_url(model_urls[arch]))

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

train_model(model,optimizer,criterion,dataloaders,
n_classes = n_classes, n_hidden=n_hidden, drop_p=drop_p,
learn_rate=learning_rate,cat_to_name = cat_to_name, n_epochs=n_epochs,
device = device ,save_dir = save_dir, arch = arch)
