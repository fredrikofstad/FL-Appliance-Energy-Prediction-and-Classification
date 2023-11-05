import pandas as pd
from sklearn.model_selection import train_test_split


data_path = "./data/"
intervals = 96  # amount of quarter hours in a day
appliance_mapping = {}
appliance_names = pd.read_csv(data_path+"Consumer1.csv").columns[2:].tolist()
for i, appliance in enumerate(appliance_names):
    appliance_mapping[appliance] = i


def read_to_df(index: int):
    """
    Creates pandas dataframe of the chosen index
    :param index: int - A value between 1 and 50
    :return: pandas dataframe
    """
    return pd.read_csv(data_path+"Consumer" + str(index) + ".csv")


def sum_energy_consumption(dataframe):
    """
    sums all the appliances' energy in a column and adds it into a total consumtion column.
    :param dataframe: a pandas dataframe
    :return: pandas dataframe
    """
    # Columns for appliances start from the 3rd column (index 2)
    appliance_columns = dataframe.columns[2:]
    # Sum the energy consumption for the household
    dataframe['Total Consumption'] = dataframe[appliance_columns].sum(axis=1)

    return dataframe


def create_io_pairs(dataframe, days: int = 7):
    """
    Creates a list of an input consisting of a certain amount of days,
    and an output of the next day. (ex: input: days 1-7 output: day 8)
    :param dataframe: A pandas dataframe
    :param days: int, amount of days
    :return: list of input output pairs
    """
    input_duration = days * intervals
    input_output_pairs = []

    for i in range(len(dataframe) - input_duration):
        input_data = dataframe['Total Consumption'].iloc[i:i+input_duration].values
        output_data = dataframe['Total Consumption'].iloc[i+input_duration]
        input_output_pairs.append((input_data, output_data))

    return input_output_pairs


def create_dataset(dataframe):
    appliance_time_series = {appliance: [] for appliance in appliance_names}

    for appliance in appliance_names:

        appliance_data = dataframe[['Periods', appliance]]
        daily_data = appliance_data.groupby(appliance_data.index // intervals).sum()
        appliance_time_series[appliance] = daily_data[appliance].tolist()

    appliance_labels = []

    for appliance in appliance_names:
        appliance_labels += [appliance] * len(appliance_time_series[appliance])

    # Split the data into training and test sets
    data = [data_point for appliance_data in appliance_time_series.values() for data_point in appliance_data]

    return train_test_split(data, appliance_labels, test_size=0.2, random_state=5460)


def get_appliance_names():
    return appliance_names


if __name__ == "__main__":
    df = read_to_df(4)
    df_sum = sum_energy_consumption(df)
    x_train, x_test, y_train, y_test = create_dataset(df_sum)
    print(y_test)

