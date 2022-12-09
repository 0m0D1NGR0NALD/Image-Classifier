import torch
from torch import nn, optim

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import numpy as np
import json
from PIL import Image

import argparse

arch = {"vgg16":25088,"densenet121":1024}

parser = argparse.ArgumentParser(description='Parser for predict.py')

parser.add_argument('--input',default='./flowers/test/1/image_06752.jpg',nargs='?',action="store",type=str)
parser.add_argument('--dir',action="store",dest="data_dir",default="./flowers/")
parser.add_argument('--checkpoint',default='./checkpoint.pth',nargs='?',action="store",type=str)
parser.add_argument('--top_k',default=5,dest="top_k",action="store",type=int)
parser.add_argument('--category_names',dest="category_names",action="store",default='cat_to_name.json')
parser.add_argument('--gpu',default="gpu",action="store",dest="gpu")

args = parser.parse_args()

path_image = args.input
number_of_outputs = args.top_k
device = args.gpu
path = args.checkpoint

def process_image(image):
    # Processing a PIL image to use in a the model
    img = Image.open(image)
    img_transforms = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                        ])
    image = img_transforms(img)
    
    return image

def predict(image_path,model,topk=5,device='gpu'):   
    model.to('cuda')
    model.eval()
    img = process_image(image_path)
    img = img.numpy()
    img = torch.from_numpy(np.array([img])).float()

    with torch.no_grad():
        output = model.forward(img.cuda())
        
    probability = torch.exp(output).data
    
    return probability.topk(topk)

def setup_network(structure='vgg16',dropout=0.2,hidden_units=4096,lr=0.001,device='gpu'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  
    if structure == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif structure == 'densenet121':
        model = models.densenet121(pretrained=True)
  
    # Defining a new and untrained feed-forward network as a classifier, using ReLU activation and Dropout
    for param in model.parameters():
        param.requires_grad = False    

    model.classifier = nn.Sequential(nn.Linear(arch['vgg16'],hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(dropout),
                                     nn.Linear(hidden_units,102),
                                     nn.LogSoftmax(dim=1)
                                    )
    model = model.to('cuda')
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr)

    return model


def load_checkpoint(path='checkpoint.pth'):
    checkpoint = torch.load(path)
    lr = checkpoint['learning_rate']
    hidden_units = checkpoint['hidden_units']
    dropout = checkpoint['dropout']
    epochs = checkpoint['no_of_epochs']
    structure = checkpoint['structure']

    model = setup_network(structure,dropout,hidden_units,lr)
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

def main():
    model = load_checkpoint(path)
    with open('cat_to_name.json', 'r') as json_file:
        cat_to_name = json.load(json_file)
        
    probabilities = predict(path_image,model,number_of_outputs,device)
    labels = [cat_to_name[str(idx + 1)] for idx in np.array(probabilities[1][0])]
    probability = np.array(probabilities[0][0])
    i = 0
    while i < number_of_outputs:
        print("Class: {} | Probability: {}".format(labels[i].title(), probability[i]))
        i += 1
    print("Prediction Complete.")

    
if __name__== "__main__":
    main()