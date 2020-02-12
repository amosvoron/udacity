import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Message', engine)
metrics = pd.read_sql_table('Metrics', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# Note that in the controllers (.index and .go methods) 
# we will not prepare entire graphs, just data that will be passed
# further to the View part of the code (master template) 
# where the graphs will be finally created.

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # data 1: Count of message by genres
    genre_counts = df.groupby('genre').count()['message']
    data1 = {
        'genre_names': list(genre_counts.index),
        'genre_counts': df.groupby('genre').count()['message']
    }

    # data 2: Count of messages by categories
    data2 = {
        'category_names': df.iloc[:,4:].columns,
        'category_counts': (df.iloc[:,4:] != 0).sum().values  
    }

    # data 3: Count of messages by categories (for histogram)
    data3 = {
        'category_counts': (df.iloc[:,4:] != 0).sum().values  
    }    
    
    data = [data1, data2, data3]
    
    # set data_type as 'general' in order to distinguish it 
    # from the 'classifier' data in the master template
    data_type = 'general'
           
    # encode data in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(data)]
    dataJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with data for plotly graphs
    return render_template(
        'master.html', 
        data_type = data_type,
        ids=ids, 
        data=dataJSON,
        classification_result=[])


# web page that handles user query and displays model results
@app.route('/go')
def go():
    
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))
  
    # data 1: metrics
    data1 = {
        'category_names': list(metrics['category']),
        'accuracy': list(metrics['accuracy'])  
    }       
    
    data = [data1]
    
    # set data_type as 'classifier' in order to distinguish it 
    # from the 'general' data in the master template
    data_type = 'classifier'
    
    # encode data in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(data)]
    dataJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)  
    
    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results,
        data_type = data_type,
        ids=ids,
        data=dataJSON
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()