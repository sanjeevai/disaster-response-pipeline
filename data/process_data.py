# import statements
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import argparse

parser = argparse.ArgumentParser(description="Takes two CSV files as inputs, \
                                cleans them and exports the saved dataset to an \
                                SQLite database")

# add argument for message CSV file
parser.add_argument('--msg',
                    type=str,
                    help='Full path of messages data CSV file')

# add argument for categories CSV file
parser.add_argument('--cat',
                    type=str,
                    help='Full path of categories data CSV file')

# add argument for SQL database file
# parser.add_argument('--database_file_path',
#                     type=str,
#                     help='Full of the database(.db) file which will contain the\
#                      cleaned data')

args = parser.parse_args()

def load_data(messages_file_path, categories_file_path):
    """
    Args:
    messages_file_path: Full path of messages CSV file
    categories_file_path: Full path of categories CSV file

    Returns:
    merged_df: Dataframe obtained from merging the two input data
    """

    messages = pd.read_csv(messages_file_path)
    categories = pd.read_csv(categories_file_path)
    
    merged_df = messages.merge(categories, on='id')
    
    # print(merged_df.head())
    
    return merged_df



def main():
    print(">>>")
    print(">>>MERGING MESSAGES AND CATEGORIES DATA...")
    print(">>>")
    if args.msg:
        messages_file_path = args.msg
    if args.cat:
        categories_file_path = args.cat
    load_data(messages_file_path, categories_file_path)
if __name__ == '__main__':
    main()