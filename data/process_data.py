# import statements
import pandas as pd
from sqlalchemy import create_engine
import argparse

parser = argparse.ArgumentParser(description="Takes two CSV files as inputs, \
                                cleans them and exports the saved dataset to \
                                an SQLite database")

# add argument for message CSV file
parser.add_argument('--msg',
                    type=str,
                    help='Full path of messages data CSV file')

# add argument for categories CSV file
parser.add_argument('--cat',
                    type=str,
                    help='Full path of categories data CSV file')

# add argument for SQL database file
parser.add_argument('--db',
                    type=str,
                    help='Full path of the database(.db) file which will \
                    contain the cleaned data')

args = parser.parse_args()

def load_data(messages_file_path, categories_file_path):
    """
    Takes inputs as two CSV files, imports them as pandas dataframe. Combines \
    them into a single dataframe

    Args:
    messages_file_path str: Messages CSV file path
    categories_file_path str: Categories CSV file path

    Returns:
    merged_df pandas_dataframe: Dataframe obtained from merging the two input data
    """

    messages = pd.read_csv(messages_file_path)
    categories = pd.read_csv(categories_file_path)
    
    merged_df = messages.merge(categories, on='id')
    
    return merged_df

def clean_data(merged_df):
    """
    Cleans the combined dataframe for use by ML model
    
    Args:
    merged_df pandas_dataframe: Merged dataframe returned from load_data() \
    function

    Returns:
    df pandas_dataframe: Cleaned data to be used by ML model
    """

    # Split categories into separate category columns
    df = merged_df # renaming for simplicity
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
    df.drop(columns='categories', axis = 1, inplace = True)

    # concatenate the original dataframe with the new `categories` dataframe
    df[categories.columns] = categories

    # drop duplicates
    df.drop_duplicates(inplace = True)

    return df

def save_data(df, database_file_path):
    """
    Saves cleaned data to an SQL database

    Args:
    df pandas_dataframe: Cleaned data returned from clean_data() function
    database_file_path str: File path of SQL Database into which the cleaned\
    data is to be saved

    Returns:
    None
    """
    
    engine = create_engine('sqlite:///{}'.format(database_file_path)) 
    db_file_name = database_file_path.split("/")[-1] # extract file name from \
                                                     # the file path
    table_name = db_file_name.split(".")[0]
    df.to_sql(table_name, engine, index=False, if_exists = 'replace')


def main():
    print(">>> ...")
    print(">>> MERGING MESSAGES AND CATEGORIES DATA")
    print(">>> ...")
    if args.msg:
        messages_file_path = args.msg
    if args.cat:
        categories_file_path = args.cat
    if args.db:
        database_file_path = args.db
    merged_df = load_data(messages_file_path, categories_file_path)
    print(">>> DATA MERGED")
    print(">>> CLEANING DATA")
    print(">>> ...")
    df = clean_data(merged_df)
    print(">>> DATA CLEANED")
    print(">>> EXPORTING TO SQL DATABASE")
    print(">>> ...")
    save_data(df, database_file_path)
    print(">>> EXPORTED TO SQL DATABASE")
    print(">>> DATA HAS BEEN EXPORTED HERE: data/{}".format(database_file_path.split("/")[-1]))

# run
if __name__ == '__main__':
    main()