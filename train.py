import argparse
import utils 

def main():
    ''' Training application with the following parameters:
        - save_dir (required)
        - arch (architecture)
        - learning_rate (learning rate)
        - hidden_units (number of units of the first hidden layer)
        - hidden_units2 (number of units of the second hidden layer)
        - epochs (number of epochs)
        - gpu (use GPU for tranining)
        - check (for checking the model architecture, no training is done)
    '''
    
    parser = argparse.ArgumentParser(description='The neural network training application. Two architectures, densenet121 and vgg16, with two hidden layers are supported.')

    parser.add_argument("data_dir", 
                        type = str,
                        help="Directory with training, validation and testing data. Default directory: 'flowers'",
                        nargs="?", 
                        action="store", 
                        default="flowers")
    parser.add_argument("--save_dir", 
                        type = str,
                        help="Checkpoint file name where the model will be saved after the training completes. Default file name: 'checkpoint.pth'",
                        action="store", 
                        default="checkpoint.pth")    
    parser.add_argument("--arch", 
                        type = str,
                        help="Model's architecture. Use 'densenet' for densnet121 and 'vgg' for vgg16. Default architecture: 'densenet'",
                        action="store", 
                        choices=['densenet', 'vgg'],
                        default="densenet")   
    parser.add_argument("--learning_rate",
                        type = float,
                        help="Optimizer's learning rate. Default learning rate: 0.001",
                        action="store", 
                        default=0.001) 
    parser.add_argument("--dropout",
                        type = float,
                        help="Classifier's dropout. Default value: 0.5",
                        action="store", 
                        default=0.5)      
    parser.add_argument("--hidden_units",
                        type = int,
                        help="Number of hidden units of the first hidden layer. Default value: 4096",
                        action="store", 
                        default=4096) 
    parser.add_argument("--hidden_units2",
                        type = int,
                        help="Number of hidden units of the second hidden layer. If 0, the second hidden layer is omitted. Default value: 512",
                        action="store", 
                        default=512)     
    parser.add_argument("--epochs",
                        type = int,
                        help="Number of training epochs. Default value: 1",
                        action="store", 
                        default=1)     
    parser.add_argument("--eval_steps",
                        type = int,
                        help="Number of training steps that have to pass to perform the validation step. Default value: 25",
                        action="store", 
                        default=25)      
    parser.add_argument("--gpu",
                        help="Use GPU. If not specified the default device will be used.",
                        action="store_true", 
                        default=False)   
    parser.add_argument("--check", 
                        help="Creates model and displays its architecture. No training is done.",
                        action="store_true", 
                        default=False)
    
    args = parser.parse_args()
    
    print(args)
    
    # Validate
    utils.validate_positive('learning_rate', args.learning_rate)
    utils.validate_positive('dropout', args.dropout)
    utils.validate_positive('hidden_units', args.hidden_units)
    utils.validate_positive('hidden_units2', args.hidden_units2)
    utils.validate_positive('epochs', args.epochs)
    utils.validate_positive('eval_steps', args.eval_steps)
    
    # Handle device_name
    if args.gpu:
        device_name = 'gpu'
    else:
        device_name = 'cpu' 
        
    # Train
    
    print('')
    print('********************************')
    print('*** Training process started ***')
    print(f'***          ({device_name.upper()})           ***')
    print('********************************')
    print('')
    
    trainloader , validloader, testloader, train_data = utils.load_data(args.data_dir)
    
    model, criterion, optimizer = utils.create_model(
        architecture = args.arch, 
        hidden_layer1 = args.hidden_units, 
        hidden_layer2 = args.hidden_units2, 
        dropout = args.dropout, 
        learning_rate = args.learning_rate, 
        device_name = device_name)
    
    # Check the model
    if (args.check):
        print(model)
        print('')
        print('The model has been created. No training will took place in check mode!')
        print('')
        return
    
    utils.train(
        model, 
        criterion, 
        optimizer, 
        args.epochs, 
        args.eval_steps, 
        trainloader,
        validloader,
        device_name) 

    utils.test(model, criterion, testloader)
    
    utils.save_model(model, train_data, args.save_dir)
    
    print('')
    print('*********************************')
    print('*** Training process finished ***')
    print('*********************************')
    print('')
    
# ----------
# Execute
# ----------
if __name__ == '__main__': main()