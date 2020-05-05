import argparse
import json
import torch
import numpy as np
from torchvision import models

from PIL import Image

def process_image(image_path):
    
    im = Image.open(image_path)
    if im.height > im.width:
        (width,height) = 256,im.height*256//im.width
    else:
        (width,height) = im.width*256//im.height,256
    
    im = im.resize((width,height))
    left = (im.width-224)//2
    upper = (im.height-224)//2
    right = left + 224
    lower = upper + 224
    im = im.crop((left,upper,right,lower)) # crop center 224x224
    np_image = np.array(im)/255 # convert color channels to 0-1 float
    np_image = (np_image-[0.485,0.456,0.406])/[0.229,0.224,0.225]
    
    return torch.tensor(np_image.transpose())

def load_checkpoint(model_path):
    checkpoint = torch.load(model_path) # load dictionary file holding the save point
    try: 
        thismodel = getattr(models,checkpoint['arch'])(pretrained=True)
    except AttributeError : # prints an error if the specified architecture isn't supported
        print('invalid architecture')
        return None
    except : 
        print('unknown error')
        return None
    
    thismodel.class_to_idx = checkpoint['indices']
    if type(thismodel) == models.ResNet:
        thismodel.fc = checkpoint['classifier']
        print('resnet arch')
    else:
        thismodel.classifier = checkpoint['classifier']
    thismodel.load_state_dict(checkpoint['model_state_dict'])
    return thismodel


parser = argparse.ArgumentParser() # create a new argument parser
parser.add_argument('input',
                    help="path to input images") # specify path to input images 
parser.add_argument('checkpoint',
                    help="model checkpoint") # specify a checkpoint for pretrained model
parser.add_argument('--top_k',
                    type=int,
                    default=3,
                    help="specify number of top results to return") # specify 
parser.add_argument('--category_names',
                    default='cat_to_name.json',
                    help="jason mapping of categories to names")
parser.add_argument('--gpu',
                    action='store_true',
                    default=False,
                    help="gpu or nah")

args = parser.parse_args() # parse command line arguments

# import category to name mapping from json file
with open(args.category_names,'r') as f:
    cat_to_name = json.load(f)

# define the hardware device to run the analysis
if args.gpu: # if gpu is specified 
    if torch.cuda.is_available(): # check if gpu is available
        device = torch.device("cuda")
    else:
        device = torch.device("cpu") # fall back to cpu and print a warning
        print("WARNING : gpu specified but not available in hardware")
else:
    device = torch.device("cpu") # default to cpu

#pre process the image
np_img = process_image(args.input)
np_img = np_img.unsqueeze_(0)
np_img = np_img.float().to(device) # move image to the appropriate device

#load checkpoint
thismodel = load_checkpoint(args.checkpoint)
thismodel = thismodel.to(device) # move model to appropriate device

thismodel.eval() # put model in evaluation mode
with torch.no_grad(): # run without gradiant
    
    logps = thismodel(np_img) # run the image through the model
    ps = torch.exp(logps) # get the probability
    probs,classes = ps.topk(args.top_k,dim=1) # get the probability of the classes of the top K 

guessidx = []
guessnames = []

for ea in classes.tolist()[0]:
    for key,value in thismodel.class_to_idx.items():
        if value == ea:
            guessidx.append(key)
            guessnames.append(cat_to_name[key])
            
# print out the results
print("probabilities: {}".format(probs))
print("class indices : {}".format(guessidx))
print("classes: {}".format(guessnames))
