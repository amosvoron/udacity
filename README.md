# Disaster Response Pipeline Project
This is a Data Science project for Udacity Nanodegree program that combines Software Engineering and Data Engineering tasks with an objective to create a model trained to classify disaster response messages accessible to the end-user through the web application.

## Summary

While the Software Engineering part includes writing a clean, modular, well-documented python code that performs Data Engineering tasks, the Data Engineering part consists of preparing 3 types of pipelines whose input data are pre-labeled text messages from real life disasters:

1. ETL (Extract-Transform-Load) pipeline
2. NLP (Natural Language Processing) pipeline
3. ML (Machine Learning) pipeline

We use ETL pipeline to prepare input data. Then we apply NLP techniques to normalize, tokenize, and lemmatize words extracted from text messages. In the last step we use ML pipeline (including TF-IDF pipeline) to build a supervised learning model. 

Lastly, we create a simple Flask website that reads the stored model and classifies the messages passed by end-user. Apart from the classification task the application also shows some general visualization graphs of the training dataset and the graph based on the classification result.

### General Graphs

<div align="center">
  <img src="Graphs1-3.jpg">
</div>

### Classification Metrics Graph

<div align="center">
  <img src="Graph-4.jpg">
</div>

## Installation
### Clone
```sh
$ git clone https://github.com/amosvoron/Udacity_DisasterResponse
```

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

