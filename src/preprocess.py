import numpy as np
import tensorflow as tf
import torch

"""
This module prepares the data for use in tensorflow_federated models
"""


def prepare_prediction_tensors(df, chunk_size=96, window_size=7):
    """
    For the prediction model:
    processes the necessary changes to the dataset to work with tensorflow_federated
    outputs a tuple containing input, output pairs
    :param df: dataframe from data.py
    :return: input_train, output_train, input_test, output_test
    """
    df = df.iloc[:, 2:]
    df['energy_consumption'] = df.sum(axis=1)

    # calculate the total number of chunks
    num_chunks = len(df) // chunk_size

    # calculate the ranges for matrix conversion
    train_range = 288
    test_range = 359

    # Initialize an empty list to store chunks as lists
    aggregated_values = []

    # Iterate through the chunks and convert each chunk to a list
    for i in range(num_chunks):
        # Extract the chunk of data
        chunk = df.iloc[i * chunk_size: (i + 1) * chunk_size]

        # Convert the chunk to a list and append it to the list of aggregated values
        chunk_list = chunk['energy_consumption'].tolist()
        aggregated_values.append(chunk_list)

    def create_input_output_pairs(data):
        input_output_pairs = []
        input_output_pair_index = []
        total_days = len(data)
        for i in range(total_days - window_size):
            input_data = data[i:i+window_size]
            input_data_index = np.array(list(range(i, i + window_size)))
            output_data = data[i+window_size:i+window_size+1]
            output_data_index = np.array(list(range(i+window_size, i + window_size+1)))
            input_output_pairs.append((input_data, output_data))
            input_output_pair_index.append((input_data_index, output_data_index))

        return input_output_pairs, input_output_pair_index

    keys = create_input_output_pairs(aggregated_values)

    convert_tensor_input_train = []
    for j in range(train_range):
        convert_to_input_matrix = np.vstack([keys[0][j][0][i] for i in range(0, 7)])
        convert_tensor_input_train.append(convert_to_input_matrix)

    tensor_converted_input_train = tf.convert_to_tensor(convert_tensor_input_train, dtype=tf.float64)

    convert_tensor_input_test = []
    for j in range(train_range, test_range):
        convert_to_input_matrix = np.vstack([keys[0][j][0][i] for i in range(0, 7)])
        convert_tensor_input_test.append(convert_to_input_matrix)
    tensor_converted_input_test = tf.convert_to_tensor(convert_tensor_input_test, dtype=tf.float64)

    tensor_converted_input_train = tf.transpose(tensor_converted_input_train, perm=[0, 2, 1])
    tensor_converted_input_test = tf.transpose(tensor_converted_input_test, perm=[0, 2, 1])

    convert_tensor_output_train = []
    for j in range(train_range):
        convert_to_output_matrix = keys[0][j][1][0]
        convert_tensor_output_train.append(convert_to_output_matrix)

    tensor_converted_output_train = tf.convert_to_tensor(convert_tensor_output_train, dtype=tf.float64)

    convert_tensor_output_test = []
    for j in range(train_range,test_range):
        convert_to_output_matrix = keys[0][j][1][0]
        convert_tensor_output_test.append(convert_to_output_matrix)

    tensor_converted_output_test = tf.convert_to_tensor(convert_tensor_output_test, dtype=tf.float64)

    return (tensor_converted_input_train, tensor_converted_output_train,
            tensor_converted_input_test, tensor_converted_output_test)


def prepare_classification_tensors(df):
    """
    For the classification model:
    processes the necessary changes to the dataset to work with tensorflow_federated
    outputs a tuple containing training, and testing data
    :param df: dataframe from data.py
    :return: input_train, output_train, input_test, output_test
    """
    df = df.iloc[:, 2:]
    num_full_days = len(df) // 96

    # Split the DataFrame into chunks of 96 rows each
    daily_chunks = [df.iloc[i * 96 : (i + 1) * 96] for i in range(num_full_days)]

    import numpy as np

    # Function to convert a DataFrame to a matrix and add a row at the top
    def convert_to_matrix_with_row(df):
        # Convert the DataFrame to a matrix
        data_matrix = df.values

        # Add a row at the top containing integers from 1 to 10
        extra_row = np.arange(1, 11).reshape(1, -1)

        # Concatenate the extra row and the data matrix vertically
        data_matrix_with_row = np.concatenate((extra_row, data_matrix), axis=0)
        return data_matrix_with_row.T

    # Convert each DataFrame in daily_chunks to a matrix with an additional row
    matrices_with_rows = [convert_to_matrix_with_row(df) for df in daily_chunks]

    # Generate labels (integer values from 0 to 9)
    labels = np.arange(10)

    # Repeat labels for each day
    labels_concat = np.tile(labels, num_full_days)

    matrices_with_rows_concat = np.vstack(matrices_with_rows)

    num_rows = matrices_with_rows_concat.shape[0]

    # Generate a random shuffling order for the rows
    shuffled_indices = np.arange(num_rows)
    np.random.shuffle(shuffled_indices)

    # Shuffle rows in both matrices based on the same shuffling order
    input = matrices_with_rows_concat[shuffled_indices]
    output = labels_concat[shuffled_indices]
    input = input[:, 1:]

    input = torch.tensor(input).unsqueeze(1)
    output = torch.tensor(output)

    input_train = input[:2300]
    output_train = output[:2300]
    input_test = input[2300:]
    output_test = output[2300:]

    input_train = torch.tensor(input_train)
    input_test = torch.tensor(input_test)

    output_train = torch.tensor(output_train)
    output_test = torch.tensor(output_test)

    return input_train, output_train, input_test, output_test
