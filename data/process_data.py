import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Loads messages and categories, merges them and returns the merged dataset.

    Input:
        messages_filepath: file path of the messages CSV file
        categories_filepath: file path of categories CSV file
    Output:
        DataFrame object containing the merged dataset
    '''
    # load messages
    df_messages = pd.read_csv(messages_filepath)
    # load categories
    df_categories = pd.read_csv(categories_filepath)
    # merge both datasets (join by id)
    df = df_messages.merge(df_categories, left_on='id', right_on='id')
    
    # return merged dataset
    return df


def clean_data(df):
    '''
    Cleans data by spliting the categories column into separate, clearly named columns, 
    converts values to binary, and drops duplicates.
    
    Input:
        df: DataFrame object containing dataset to clean
    Output:
        DataFrame object with cleaned dataset    
    '''
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]
    # use row to extract clean column names (trimming last 2 characters)
    category_colnames = row.str.extract(r'(.*)-.', expand=False)
    # assign clean column names to the columns
    categories.columns = category_colnames
    
    # convert values to binary (using the last character)
    for column in categories:
        categories[column] = categories[column].str.extract(r'(\d)$', expand=False)    
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from df
    df.drop('categories', axis=1, inplace=True)
    # merge messages with categories (join by id)  
    categories['id'] = df['id']
    df = df.merge(categories, left_on='id', right_on='id')
    
    # remove duplicates
    df = df.drop_duplicates()
    
    # return cleaned dataset
    return df

    
def save_data(df, database_filename, table_name='Message'):
    '''
    Stores the clean data into a SQLite database.
    
    Input:
        df: DataFrame data to save
        database_filename: SQLite database name
        table_name: messages data table name (optional)
    Output:
        None    
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql(table_name, engine, if_exists='replace', index=False)  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)       
        print('Loaded data shape: {}'.format(df.shape))

        print('Cleaning data...')
        df = clean_data(df)
        print('Cleaned data shape: {}'.format(df.shape))
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()