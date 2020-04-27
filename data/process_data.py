"""
Preprocessing data
Disaster Response Pipeline Project
Udacity - Data Scientist Nanodegree
Sample Script Execution:
> python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db
Arguments:
    1) Messages in csv file format (disaster_messages.csv)
    2) Categories in csv file format (disaster_categories.csv)
    3) SQLite destination database (DisasterResponse.db)
"""

import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
	"""
    Load Data function
    
    Inputs:
        messages_filepath -> path to messages csv file
        categories_filepath -> path to categories csv file
    
	Output:
        df -> Loaded data as Pandas DataFrame after merging both
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id',how='inner')

    return df


def clean_data(df):
	"""
    Clean Data function
    
    Arguments:
        df -> raw data Pandas DataFrame
    Outputs:
        df -> clean data Pandas DataFrame . Removes url , common words , duplicates , punctuations and converts ot lower case
    """
	
    categories = df.categories.str.split(';', expand=True)

    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x: x[:-2])

    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x[-1])
        categories[column] = pd.to_numeric(categories[column])
    
    df = df.drop(columns= 'categories', axis = 1)
    
    df = pd.concat([df, categories], axis = 1)
    
    df=df.drop_duplicates(subset='id', keep = False)
    

    return df


def save_data(df, database_filename):
	"""
    Save Data function
    
	Saves the Dataframe into db file format
    Arguments:
        df -> Clean data Pandas DataFrame
        database_filename -> database file (.db) destination path
    """
    engine = create_engine('sqlite:///'+database_filename)
    table_name=database_filename[database_filename.rfind("/")+1:-3]
    print("\tTable Name is",table_name)
    df.to_sql(table_name, engine, index=False,if_exists='replace')
    

def main():
	"""
    Main Data Processing function
    
    This function implement the ETL pipeline:
        1) Data extraction from .csv
        2) Data cleaning and pre-processing
        3) Data loading to SQLite database
    """

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
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