import torch
from torch import nn,optim

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import json

import argparse

arch = {"vgg16":25088,"densenet121":1024}

parser = argparse.ArgumentParser(description='Parser for train.py')
parser.add_argument('--data_dir',action="store",default="./flowers/")
parser.add_argument('--save_dir',action="store",default="./checkpoint.pth")
parser.add_argument('--arch',action="store",default="vgg16")
parser.add_argument('--learning_rate',action="store",type=float,default=0.01)
parser.add_argument('--hidden_units',action="store",dest="hidden_units",type=int,default=512)
parser.add_argument('--epochs',action="store",default=3,type=int)
parser.add_argument('--dropout',action="store",type=float,default=0.5)
parser.add_argument('--gpu',action="store",default="gpu")

args = parser.parse_args()

data = args.data_dir
path = args.save_dir
lr = args.learning_rate
struct = args.arch
hidden_units = args.hidden_units
power = args.gpu
epochs = args.epochs
dropout = args.dropout

def load_data(root = "./flowers"):
    with open('cat_to_name.json', 'r') as file:
        cat_to_name = json.load(file)
    
    data_dir = root
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Defining transforms for the train, validation, and test sets
    train_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                          ])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                          ])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                                         ])

    # Loading the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir,transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir,transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir,transform=test_transforms)

    # Defining the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data,batch_size=64,shuffle=False)
    testloader = torch.utils.data.DataLoader(test_data,batch_size=64,shuffle=False)

    return trainloader,validloader,testloader,train_data

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
                                     nn.Linear(hidden_units, 102),
                                     nn.LogSoftmax(dim=1)
                                    )
    model = model.to('cuda')
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),lr)

    return model,criterion  

def main():
    trainloader,validloader,testloader,train_data = load_data(data)
    model, criterion = setup_network(struct,dropout,hidden_units,lr,power)
    optimizer = optim.Adam(model.classifier.parameters(),lr= 0.001)
    
    # Training model
    steps = 0
    running_loss = 0
    print_every = 5
    epochs = 1
    print("Training In Progress... ")
    for epoch in range(epochs):
        for images, labels in trainloader:
            steps += 1
            
            # Moving input and label tensors to the default device
            images, labels = images.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward pass
            logps = model.forward(images)
            loss = criterion(logps,labels)
            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to('cuda'), labels.to('cuda')

                        logps = model.forward(images)
                        loss = criterion(logps,labels)
                        valid_loss += loss.item()

                        # Calculating accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1,dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}... "
                      f"Loss: {running_loss/print_every:.3f}... "
                      f"Validation Loss: {valid_loss/len(validloader):.3f}... "
                      f"Accuracy: {accuracy/len(validloader):.3f}")
                running_loss = 0
                model.train()
    
    model.class_to_idx = train_data.class_to_idx
    torch.save({'structure' :struct,
                'hidden_units':hidden_units,
                'dropout':dropout,
                'learning_rate':lr,
                'no_of_epochs':epochs,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                path)
    
    print("Checkpoint Saved.")
if __name__== "__main__":
    main()