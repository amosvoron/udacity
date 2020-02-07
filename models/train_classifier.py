import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# ML classifiers
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV

import pickle


def load_data(database_filepath, table_name='Message'):
    '''
    Loads data from the SQLite database.

    Input:
        database_filepath: SQLite database name
    Output:
        X: input vector of messages 
        y: output vector of categories (labels)
        category_names: categories
    '''  
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table(table_name, engine)
    
    # temp: REMOVE IN PRODUCTION
    # ----------------------------------------------------
    #df = df.sample(n=1000, axis=0, random_state=0)
    # ----------------------------------------------------
    
    X = df['message'].values 
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = y.columns.values
    
    return X, y, category_names


def tokenize(text):
    '''
    Normalizes, lemmatizes, and tokenizes text.
    '''
    # normalization
    text = re.sub(r'[^a-z0-9]', ' ', text.lower())
    
    # tokenization
    words = word_tokenize(text)

    # lemmatizing
    words = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    # Stop Word removal
    words = [w for w in words if w not in stopwords.words('english')]
    
    return words


def build_model():
    '''
    Builds the model with RandomForestClassifier classifier using the parameters
    that were previously tuned by GridSearchCV. 
    
    Remark:
    This model is the winning model chosen among 4 models that were examinated 
    in the ML pipeline preparation phase. 
    (Please check the ML Pipeline Preparation.ipynb file for details.)
    '''    
    model = Pipeline([
        ('vect', CountVectorizer(
            tokenizer=tokenize, 
            ngram_range=(1,2), 
            max_df=1.0, 
            min_df=2)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(
            random_state = 0,
            n_estimators = 100,
            max_features = 'auto',
            min_samples_leaf = 1,
            min_samples_split = 2,
            bootstrap = True
        )))
    ])    
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Displays the results using accuracy, F1 score, precision and recall
    for each output category.
    ''' 
    Y_pred = model.predict(X_test)
    
    for i, col in enumerate(category_names): 
        print(f'Category: {col}')
        y_true = Y_test.iloc[:,i]
        y_pred = Y_pred[:,i]
        accuracy = (y_true == y_pred).mean()
        
        print('Accuracy: {:.4f}'.format(accuracy))
        print('Classification report:')
        print(classification_report(y_true, y_pred))
        print('--------------------------------------------------------')
        print()
        
    # total accuracy
    total_accuracy = (Y_pred == Y_test).mean().sum()/len(Y_test.columns)          
    print('Average accuracy: {:.4f}'.format(total_accuracy))
    print('--------------------------------------------------------')


def save_model(model, model_filepath):
    '''
    Serializes model and saves it as a file.
    
    Input:
        model: model to save
        model_filepath: path of a model file
    '''
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()