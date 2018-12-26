# imports
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_file_path, categories_file_path):
    """
    - Takes inputs as two CSV files
    - Imports them as pandas dataframe.
    - Merges them into a single dataframe

    Args:
    messages_file_path str: Messages CSV file
    categories_file_path str: Categories CSV file

    Returns:
    merged_df pandas_dataframe: Dataframe obtained from merging the two input\
    data
    """

    messages = pd.read_csv(messages_file_path)
    categories = pd.read_csv(categories_file_path)
    
    df = messages.merge(categories, on='id')
    
    return df

def clean_data(df):
    """
    - Cleans the combined dataframe for use by ML model
    
    Args:
    df pandas_dataframe: Merged dataframe returned from load_data() function

    Returns:
    df pandas_dataframe: Cleaned data to be used by ML model
    """

    # Split categories into separate category columns
    categories = df['categories'].str.split(";",\
                                            expand = True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0,:].values
    
    # use this row to extract a list of new column names for categories.
    new_cols = [r[:-2] for r in row]

    # rename the columns of `categories`
    categories.columns = new_cols

    # Convert category values to just numbers 0 or 1.
    for column in categories:

        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)

    # concatenate the original dataframe with the new `categories` dataframe
    df[categories.columns] = categories

    # drop duplicates
    df.drop_duplicates(inplace = True)

    return df

def save_data(df, database_file_name):
    """
    Saves cleaned data to an SQL database

    Args:
    df pandas_dataframe: Cleaned data returned from clean_data() function
    database_file_name str: File path of SQL Database into which the cleaned\
    data is to be saved

    Returns:
    None
    """
    
    engine = create_engine('sqlite:///{}'.format(database_file_name)) 
    db_file_name = database_file_name.split("/")[-1] # extract file name from \
                                                     # the file path
    table_name = db_file_name.split(".")[0]
    df.to_sql(table_name, engine, index=False, if_exists = 'replace')


def main():
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


# run
if __name__ == '__main__':
    main()