import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_messages_and_categories(messages_filepath, categories_filepath):
    """
     Load Messages Data with Categories Function

     Arguments:
         messages_filepath -> Path to the CSV file containing messages
         categories_filepath -> Path to the CSV file containing categories
     Output:
         df -> Combined data containing messages and categories
     """

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')

    return df

def clean_data_categories(df):
    """
    Prepare the categories data to numeric type

    Arguments:
        df -> Dataframe containing the merged data from categories and messages
    Outputs:
        df -> Same Dataframe with categories column transformed to 36 numericcal columns, one for each category
    """

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';', expand=True)

    # Assign the correct name to each category column
    row = categories.iloc[0]
    category_colnames = list(row.str.split('-').str[0].values)
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # Concatenate dataframe with correct categories column and drop duplicates
    df = df.drop('categories',axis=1)
    df = pd.concat([df,categories],axis=1)
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """
     Save the clean dataset into an sqlite database.

     Arguments:
         df -> Cleaned Dataframe with messages and categories
         database_filename -> Path to SQLite destination database
    Output:
        Sqlite database of the data
    """

    engine = create_engine('sqlite:///{}.db'.format(database_filename))
    df.to_sql(database_filename, engine, index=False, if_exists='replace')




def main():
    """
    Main function which will kick off the data processing functions. There are three primary actions taken by this function:
        1) Load Messages Data with Categories
        2) Clean Categories Data
        3) Save Data to SQLite Database
    """

    # Execute the ETL pipeline if the count of arguments is matching to 4
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_messages_and_categories(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data_categories(df)
        
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