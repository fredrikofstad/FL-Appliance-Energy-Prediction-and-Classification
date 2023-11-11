import pandas as pd

data_path = "../data/"


def create_df():
    # Dictionary to store DataFrames
    dataframes = {}
    name_list = []

    # Loop through each CSV file and read it into a DataFrame
    for i in range(1, 51):
        file_name = f'Consumer{i}.csv'
        name_list.append(file_name)
        file_path = data_path + file_name

        # Read the CSV file into a pandas DataFrame and store it in the dictionary
        dataframes[file_name] = pd.read_csv(file_path)

    return dataframes, name_list
