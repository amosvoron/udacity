# Udacity_StackOverflow
This is a Data Science project for Udacity Nanodegree program based on the 2019 Stack Overflow's survey results. 
## Installation
### Clone
```sh
$ git clone https://github.com/amosvoron/Udacity_StackOverflow
```
## Project Motivation
The aim of the project is to apply CRISP-DM (Cross-Industry Standard Process for Data Mining) over the chosen data set. We'll apply all required CRISP-DM steps over our data which are as follows:
- Business Understanding
- Data Understanding
- Data Preparation
- Modeling
- Evaluation
- Deployment

We'll analyse the data by setting up three questions/topics of interest:
1. What are the most popular programming languages, databases, platforms, web frameworks, and other frameworks among the developers? 
2. Is there any strong correlation between the education level and non-degree education? Is there a notable difference between the low education and high education in relation to the non-formal education?
3. We want to predict the salary using the "moderate quantity" of variables that we'll find significant for the salary prediction. By the "moderate quantity" we mean any quantity that is substantially lower than the one from the Udacity lesson case where more than 1000 variables were used reaching the test R2 score about 0.7. Our goal will be to obtain similar result with less variables.
## File Descriptions
- **Stackoverflow.ipynb**: Jupyter Notebook document with python code
- **survey_result_public.zip**: data set (zipped containing the CSV file)
- **survey_result_schema.csv**: data set's schema (in CSV format)
- **schema.csv**: data set's schema with Section column
- **so_survey_2019.pdf**: survey's questions
## License
MIT