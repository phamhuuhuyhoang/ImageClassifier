import numpy as np
import torch
from torch import nn,optim
from torchvision import datasets,transforms,models
import argparse

def load_model(arch):
    try:
        thismodel = getattr(models,arch)(pretrained=True)
    except AttributeError:
        print('invalid architecture, defaulting to vgg19')
        thismodel = models.vgg19(pretrained=True)
    # keep the pretrained parameters
    for param in thismodel.parameters():
        param.requires_grad=False    
    # print progress
    print("setting up a classifier with {} hidden units".format(args.hidden_units))
    # match the in features with the architecture of the pretrained model
    if type(thismodel) == models.VGG:
        in_features = thismodel.classifier[0].in_features
    elif type(thismodel) == models.AlexNet:
        in_features = thismodel.classifier[1].in_features
    elif type(thismodel) == models.DenseNet:
        in_features = thismodel.classifier.in_features
    elif type(thismodel) == models.ResNet:
        in_features = thismodel.fc.in_features
 
    else:
        print('unsupported architecture, default to vgg19')
        thismodel = models.vgg19(pretrained = True)
        in_features = thismodel.classifier[0].in_features
  
    # define the classifier
    classifier = nn.Sequential(nn.Linear(in_features,args.hidden_units),
                                nn.ReLU(),
                                nn.Dropout(p=0.3),
                                nn.Linear(args.hidden_units,102),
                                nn.LogSoftmax(dim=1))
    if type(thismodel) == models.ResNet or type(thismodel) == models.Inception3:
        thismodel.fc = classifier
        optim_param = thismodel.fc.parameters()
    else:
        thismodel.classifier = classifier # point the model classifier to the custom defined classifier
        optim_param = thismodel.classifier.parameters()
    return thismodel,optim_param

parser = argparse.ArgumentParser()
parser.add_argument('data_dir',
                    action='store',
                    help="directory where the training and validation data lives")
parser.add_argument('--learning_rate',
                    type=float,
                    default=0.001,
                    help="learning rate")
parser.add_argument('--arch',
                    default = 'vgg19',
                    help="torchvision architecture")
parser.add_argument('--save_dir',
                    default = '',
                    help = "directory to save a checkpoint")
parser.add_argument('--hidden_units',
                    type=int,
                    default = 512,
                    help = "hidden unit in classifier")
parser.add_argument('--epochs',
                    type=int,
                    default = 1,
                    help="number of epochs to train")
parser.add_argument('--gpu',
                    action='store_true',
                    help="gpu or nah",
                    default=False)
args = parser.parse_args() # parse argument


# identify the training directory
train_dir = args.data_dir + '/train' # training directory
valid_dir = args.data_dir + '/valid' # validation directory
# print progress
print("defining training/validation data located in {}".format(train_dir))
# define train transforms
train_transforms = transforms.Compose([transforms.CenterCrop(224),
                                       transforms.RandomRotation(180),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485,0.456,0.406],
                                                            [0.229,0.224,0.225])])
valid_transforms = transforms.Compose([transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485,0.456,0.406],
                                                            [0.229,0.224,0.225])])
# define the training data 
train_data = datasets.ImageFolder(train_dir,transform=train_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform = valid_transforms)
# define the data loader based on the training data
trainloader = torch.utils.data.DataLoader(train_data,batch_size=64,shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data,batch_size=64)
# check the architecture

print("Model Architecture: {}".format(args.arch))
# define the model using pretrained torch vision model
model,optim_param = load_model(args.arch) # returns the model definition and the model parameters for optimization
#model = models.vgg19(pretrained=True)

# define the hardware device to run the analysis
if args.gpu: # if gpu is specified 
    if torch.cuda.is_available(): # check if gpu is available
        device = torch.device("cuda")
        print("gpu available and enabled")
    else:
        device = torch.device("cpu") # fall back to cpu and print a warning
        print("WARNING : gpu specified but not available in hardware")
else:
    device = torch.device("cpu") # default to cpu
    print("running on cpu")

# print progress
print("settin up loss and optimizer")
criterion = nn.NLLLoss() # use negative log likelyhood loss
optimizer = optim.Adamax(optim_param,lr=args.learning_rate) # use Adamax

model.to(device) # move model to appropriate hardware


epochs = args.epochs # define how many epochs to train
# print progress
print("training....")
for ea in range(epochs):
    model.train() # put model in training mode
    running_loss = 0
    for inputs,labels in trainloader:
        inputs,labels = inputs.to(device),labels.to(device)
        optimizer.zero_grad()
        logps = model(inputs)
        loss = criterion(logps,labels)
        loss.backward()
        optimizer.step()
        running_loss+=loss.item()
    print("training loss:{}".format(running_loss/len(trainloader)))
    
    model.eval() # evaluation mode
    with torch.no_grad():
        validloss = 0
        for inputs,labels in validloader:
            inputs,labels = inputs.to(device),labels.to(device)
            logps = model(inputs)
            validloss += criterion(logps,labels).item()
        print("validation loss:{}".format(validloss/len(validloader)))
        
# printing progress
print("saving a checkpoint")
# save a checkpoint    
checkpoint = {'arch' : args.arch,
              'indices':train_data.class_to_idx,
              'model_state_dict':model.state_dict()}
if type(model) == models.ResNet:
    checkpoint['classifier'] = model.fc
else:
    checkpoint['classifier'] = model.classifier
    
torch.save(checkpoint,args.save_dir+'checkpoint.pth')