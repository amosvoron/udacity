import argparse
import utils 

def main():
    ''' Predicting application with the following parameters:
        - image (image to predict)
        - checkpoint (checkpoint file)
        - top_k (top K classes)
        - category_names (JSON file with categories)
        - gpu (use GPU for prediction)
    '''

    parser = argparse.ArgumentParser(description='The neural network prediction application.')
    
    '''
    parser.add_argument("image", 
                        type = str,
                        help="Path name of the image to predict. Default image path: 'flowers/test/17/image_03830.jpg'",
                        #nargs="?", 
                        action="store", 
                        default="flowers/test/17/image_03830.jpg")         
    '''        
    parser.add_argument("image", 
                        type = str,
                        help="Path name of the image to predict.",
                        action="store")  
    parser.add_argument("checkpoint", 
                        type = str,
                        help="Checkpoint file with the saved model. Default file name: 'checkpoint.pth'",
                        nargs="?", 
                        action="store", 
                        default="checkpoint.pth")    
    parser.add_argument("--top_k",
                        type = int,
                        help="Top K classes. Default value: 5",
                        action="store", 
                        default=5) 
    parser.add_argument("--category_names",
                        type = str,
                        help="JSON file with category names. Default value: 'cat_to_name.json'",
                        action="store", 
                        default="cat_to_name.json")        
    parser.add_argument("--gpu",
                        help="Use GPU. If not specified the default device will be used.",
                        action="store_true", 
                        default=False)   
    
    args = parser.parse_args()
    
    print(args)
    
    # Validate
    utils.validate_positive('top_k', args.top_k, True)

    # Handle device_name
    if args.gpu:
        device_name = 'gpu'
    else:
        device_name = 'cpu' 
    
    print('')
    print('**********************************')
    print('*** Prediction process started ***')
    print(f'***           ({device_name.upper()})            ***')
    print('**********************************')
    print('')    
    
    # Load model, predict & display probabilities
    model = utils.load_model(args.checkpoint, device_name)   
    probs, classes = utils.predict(args.image, model, args.top_k, device_name)
    utils.display(probs, classes, args.category_names)
    
    print('')
    print('***********************************')
    print('*** Prediction process finished ***')
    print('***********************************')
    print('')


# ----------
# Execute
# ----------
if __name__ == '__main__': main()