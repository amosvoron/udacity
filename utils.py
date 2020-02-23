#import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from workspace_utils import active_session
from collections import OrderedDict 
from PIL import Image
import json


def validate_positive(arg_name, value, greater_than_zero = False):
    if value < 0:
        raise Exception(f"A non-negative value for the argument '{arg_name}' ({value}) is required.")
        
    if greater_than_zero and value == 0:
        raise Exception(f"A value greater than zero for the argument '{arg_name}' ({value}) is required.")
        
    return


def get_device(device_name = None):
    ''' Checks if the specified device exists and acts accordingly. 
        If device is not specified (None), returns the default device.
        Returns the specified device or 'cpu' if 'gpu' is not supported.
    '''
    
    # Return default device (if not specified)
    if device_name == None:
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Check arguments
    if device_name not in ['cpu', 'gpu']:
        raise Exception(f'The specified device ({device_name}) is invalid.')
    
    # CPU is always supported
    if device_name == 'cpu':
        return torch.device("cpu")
    
    # Check GPU
    if not torch.cuda.is_available():
        print('GPU is not supported. CPU will be used.')
        return torch.device("cpu") 
    
    return torch.device("cuda:0")


def get_categories(json_file):
    ''' Returns the flower categories (from JSON file). 
    '''
    
    with open(json_file, 'r') as f:
        return json.load(f)
    

def load_data(data_dir):
    ''' Loads data with transformers for training, validation and testing. 
        Returns all three loaders.
    '''
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]) 

    valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]) 
    
    
    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=50, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=25, shuffle = False)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=25, shuffle = False)
    
    return trainloader , validloader, testloader, train_data


def create_model(architecture, hidden_layer1 = 4096, hidden_layer2 = 512, dropout=0.5, learning_rate=0.001, device_name='gpu'):
    ''' Creates model supporting vgg16 and densenet121 architecture. 
        Returns the model. 
    '''
    
    if architecture.lower() == 'vgg':
        model = models.vgg16(pretrained=True)
        n_input = 25088
    elif architecture.lower() == 'densenet':
        model = models.densenet121(pretrained=True)
        n_input = 1024
    else: 
        raise Exception("Unknown architecture. Please use 'vgg' or 'densenet' architecture.")
    
    # Freeze parameters so we don't backpropagate through them
    for param in model.parameters():
        param.requires_grad = False
    print('Grad parameters freezed.')
    
    # Define classifier
    n_output = 102
    classifier = nn.Sequential()
    classifier.add_module("input_layer", nn.Linear(n_input, hidden_layer1, bias=True))
    classifier.add_module("bn1", nn.BatchNorm1d(hidden_layer1))
    classifier.add_module("relu1", nn.ReLU())
    classifier.add_module("dropout1", nn.Dropout(dropout))       
    
    # only one hidden layer is included
    if (hidden_layer2 == 0):
        classifier.add_module("hidden_layer1", nn.Linear(hidden_layer1, n_output, bias=True))
    # add two hidden layers
    else:
        classifier.add_module("hidden_layer1", nn.Linear(hidden_layer1, hidden_layer2, bias=True))
        classifier.add_module("bn2", nn.BatchNorm1d(hidden_layer2))
        classifier.add_module("relu2", nn.ReLU())
        classifier.add_module("dropout2", nn.Dropout(dropout)) 
        classifier.add_module("hidden_layer2", nn.Linear(hidden_layer2, n_output, bias=True))
        
    # output        
    classifier.add_module("output", nn.LogSoftmax(dim=1))    
    model.classifier = classifier
                          
    # create criterion & optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    device = get_device(device_name)
    model.to(device)        
             
    print(f'Model ({architecture}) created.')
    
    return model, criterion, optimizer


def evaluate(model, criterion, loader, device_name='gpu'):
    ''' Evaluates the model calculating loss and accuracy. 
        Returns loss and accuracy.
    '''
    
    device = get_device(device_name)
    eval_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            
            batch_loss = criterion(logps, labels)       
            eval_loss += batch_loss.item()
            
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    return eval_loss, accuracy


def train(model, 
          criterion, 
          optimizer, 
          epochs, 
          eval_every_step, 
          trainloader,
          validloader,
          device_name='gpu'):
    ''' Trains the model. '''
    
    print('Training started...')
    
    device = get_device(device_name)
    steps = 0
    running_loss = 0
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            # Validate
            if steps % eval_every_step == 0:              
                test_loss, accuracy = evaluate(model, criterion, validloader, device_name)   
                     
                # Output
                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/eval_every_step:.3f}.. "
                    f"Validation loss: {test_loss/len(validloader):.3f}.. "
                    f"Validation accuracy: {accuracy/len(validloader):.3f}")                  
                
                running_loss = 0
                model.train()
            
    print('Training finished.')
    

def test(model, criterion, testloader):
    ''' Tests the model. '''
    
    print('Testing started...')
    
    test_loss, accuracy = evaluate(model, criterion, testloader)

    print(f"Testing loss: {test_loss/len(testloader):.3f}.. "
          f"Testing accuracy: {accuracy/len(testloader):.3f}")

    
def save_model(model, train_data, checkpoint_file):
    ''' Saves the model. '''
    
    print('Model checkpoint is saving...')
    
    model.class_to_idx = train_data.class_to_idx
    checkpoint = {'architecture': model.__class__.__name__,
                  'classifier': model.classifier,
                  'state_dict': model.state_dict(),
                  #'optimizer': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx}
    torch.save(checkpoint, checkpoint_file)
    
    print(f'Model checkpoint saved ({checkpoint_file}).')
    
   
def load_model(checkpoint_file, device_name):
    ''' Loads the model, 
        returns the model and criterion (non-parameterized).
    '''
    
    print(f'Model loading from {checkpoint_file} ...')
    
    #Perform load with the map_location parameter
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'

    checkpoint = torch.load(checkpoint_file, map_location=map_location)
    
    # Create pretrained model (architecture dependent)
    architecture = checkpoint['architecture']
    if architecture.lower() == 'densenet':
        model = models.densenet121(pretrained=True)
    elif architecture.lower() == 'vgg':
        model = models.vgg16(pretrained=True)
    else:
        raise Exception(f'Unknown architecture {architecture}. Cannot restore the model.')
    
    # Update model
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])   
    
    # Freeze parameters 
    for param in model.parameters():
        param.requires_grad = False
   
    # Make sure that model is set to evaluation mode
    model.eval()
    
    # Move model to current device to avoid evaluation problems with incompatible types
    model.to(get_device(device_name))

    print(f'Model with {architecture} architecture loaded from the checkpoint ({checkpoint_file}).')
    
    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model.
        Returns PIL image and tensor image.
    '''
    
    preprocess = transforms.Compose([
                               transforms.Resize(256),
                               transforms.CenterCrop(224),
                               transforms.ToTensor(),
                               transforms.Normalize(
                                               [0.485, 0.456, 0.406],
                                               [0.229, 0.224, 0.225])])
    
    img_pil = Image.open(image)
    img_tensor = preprocess(img_pil)
    
    return img_tensor


def predict(image_path, model, topk, device_name):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    #model.class_to_idx = train_data.class_to_idx
    #print(model.class_to_idx)
    
    img_tensor = process_image(image_path) 
    reshaped = img_tensor.unsqueeze(0)
    
    # handle device
    device = get_device(device_name)
    
    if (str(device) == 'cpu'):
        with torch.no_grad():
            output=model.forward(reshaped)
    else:
        with torch.no_grad():
            output=model.forward(reshaped.cuda())
    
    ps = torch.exp(output)
    probs, clss = ps.topk(topk, dim=1)
    
    # Invert classes
    inverted = {v: k for k, v in model.class_to_idx.items()}
    classes = [inverted[c] for c in clss[0].cpu().numpy()]

    return probs, classes


def display(probs, classes, category_names):
    """ Displays probabilities with flower names of the top K classes. 
    """
    
    # load categories
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    # get top names    
    max_index = np.argmax(probs)
    max_probability = probs[max_index]
    names = [cat_to_name[c] for c in classes]
    
    # output
    print('')
    print(f'Top {len(classes)} probabilities:')
    print('')
    i = 1
    for name, prob in zip(names, probs[0].cpu().numpy()):
        print(f'({i}) {name}:', '{0:.2f}%'.format(prob * 100))
        i += 1

