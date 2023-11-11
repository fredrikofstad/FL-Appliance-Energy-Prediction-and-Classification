import numpy as np
import tensorflow as tf

"""
Prepare tensors takes in a dataframe and optional parameters chunk_size and window_size,
and returns tensors for use in tensorflow, for the input_output pairs used in energy prediction
"""

def prepare_tensors(df, chunk_size=96, window_size=7):
    df = df.iloc[:, 2:]
    df['energy_consumption'] = df.sum(axis=1)

    # calculate the total number of chunks
    num_chunks = len(df) // chunk_size

    # calculate the ranges for matrix conversion
    max_range = num_chunks - window_size

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

    convert_tensor_input = []
    for j in range(max_range):
        convert_to_input_matrix = np.vstack([keys[0][j][0][i] for i in range(0, 7)])
        convert_tensor_input.append(convert_to_input_matrix)

    tensor_converted_input = tf.convert_to_tensor(convert_tensor_input, dtype=tf.float64)
    tensor_converted_input = tf.transpose(tensor_converted_input, perm=[0, 2, 1])

    convert_tensor_output = []
    for j in range(max_range):
        convert_to_output_matrix = keys[0][j][1][0]
        convert_tensor_output.append(convert_to_output_matrix)

    tensor_converted_output = tf.convert_to_tensor(convert_tensor_output, dtype=tf.float64)

    return tensor_converted_input, tensor_converted_output
