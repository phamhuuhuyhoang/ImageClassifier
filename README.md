# ImageClassifier
image classifier using [pytorch](https://pytorch.org)
## Dependencies
- python 3.7
- numpy
## Usage
Command line  
`python train.py dir --gpu --epochs`</br>
`python predict.py image_path model_path --gpu`  </br>
A directory containing validation and training data must be available:  
dir/train/  
Each category within the train folder should be a separate folder containing training images
