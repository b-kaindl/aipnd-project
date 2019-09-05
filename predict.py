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

#functions
def build_model_from_file(filename):
    checkpoint = torch.load(filename)
    base_model = arch_dict[checkpoint['arch']]
    class_model = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(checkpoint['input_size'],checkpoint['hidden_size'],)),
    ('relu1', nn.ReLU()),
    ('fc2', nn.Linear(checkpoint['hidden_size'],checkpoint['output_size'])),
    ('dropout', nn.Dropout(checkpoint['dropout_prob'])),
    ('logsoftmax', nn.LogSoftmax(dim=1))
]))

    class_model.load_state_dict(checkpoint['state_dict'])
    base_model.classifier = class_model
    loss = checkpoint['criterion']
    optimizer = checkpoint['optimizer'].load_state_dict(checkpoint['optim_state'])
    base_model.class_to_idx = checkpoint['class_to_idx']
    return[base_model, loss, optimizer]

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    im = Image.open(image)

    min_size = min(im.size)

    if im.size[0] == min_size:
        new_size = [im.size[0], 256]
    else:
        new_size = [256, im.size[1]]

    im.thumbnail(new_size)
    width, height = im.size

    left = (256 - 224)/2
    top = (256 - 224)/2
    right = (256 + 224)/2
    bottom = (256 + 224)/2

    image = im.crop((left, top, right, bottom))

    im = np.array(im)
    im = im/255.

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    im = (im - mean) / std

    im = np.transpose(im, (2, 0, 1))

    return im

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



    model.to(device)

    model.eval()

    img_data = process_image(image_path)
    in_tensor = torch.from_numpy(img_data)
    in_tensor = in_tensor.type(torch.FloatTensor)

    images = in_tensor.to(device)
    images.unsqueeze_(0)

    output = model.forward(images)
    ps = torch.exp(output)

    class_dict = model.class_to_idx

    inv_dict =  class_dict#{v: k for k,v in class_dict.items()}

    top_p, top_classes = ps.topk(topk, dim =1)
    class_out =[inv_dict[str(clid)] for clid in top_classes.tolist()[0]]
    ps_out =top_p.tolist()[0]

    output = list(zip(class_out,ps_out))


    return output

parser = argparse.ArgumentParser()

option_dict = {
    'img_path': 'Image to perform prediction on',
    'checkpoint': 'Location for trainedmodel checkpoint',
    '--gpu' : 'Enable GPU',
    '--mapping_file' : '.json mapping ID:Name',
    '--topk' : 'show top k predictions'
}

parser.add_argument("img_path",
help=option_dict['img_path'],
type = str)



parser.add_argument("checkpoint",
help=option_dict['checkpoint'],
type = str)

parser.add_argument("--topk",
help=option_dict['--topk'],
type = int)

parser.add_argument("--gpu",
help=option_dict['--gpu'], action ='store_true')


parser.add_argument("--mapping_file",
help=option_dict['--mapping_file'], type =str)

args = parser.parse_args()

img_path = args.img_path

checkpoint = args.checkpoint

if args.mapping_file:
    with open(args.mapping_file, 'r') as f:
        cat_to_name = args.mapping_file
else:
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

if args.topk:
    topk = args.topk
else:
    topk= 5

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

    #load model (we don't need optimizer and loss)

model = build_model_from_file(checkpoint)[0]
print('Loading Model...')


output = predict(img_path, model, topk)

print(output)
