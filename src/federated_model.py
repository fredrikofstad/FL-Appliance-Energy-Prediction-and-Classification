import importlib
import tensorflow as tf
import tensorflow_federated as tff
from preprocess import *
import data
import config


def train(network="RNN"):
    # reload config incase changes were made
    importlib.reload(config)
    # get hyperparameters from config
    epochs = config.NUM_EPOCHS
    batch_size = config.BATCH_SIZE
    shuffle_buffer = config.SHUFFLE_BUFFER
    prefetch_buffer = config.PREFETCH_BUFFER
    seq_length = config.SEQ_LENGTH
    input_size = config.INPUT_SIZE
    output_size = config.OUTPUT_SIZE
    client_learning_rate = config.CLIENT_LEARNING_RATE
    server_learning_rate = config.SERVER_LEARNING_RATE

    consumer_data = {}

    dataframe, name_list = data.create_df()
    for sheet_name, df in dataframe.items():
        consumer_data[sheet_name] = df

    processed_data = [prepare_tensors(consumer_data[name_list[i]]) for i in range(1, 50)]

    input_data_list = [processed_data[i][0].numpy() for i in range(49)]
    target_data_list = [processed_data[i][1].numpy() for i in range(49)]

    num_clients = len(input_data_list)

    def create_tf_dataset(input_data, target_data):
        ds = tf.data.Dataset.from_tensor_slices((input_data, target_data))
        return ds

    client_datasets = []
    for i in range(num_clients):
        dataset = create_tf_dataset(input_data_list[i], target_data_list[i])
        dataset = dataset.repeat(epochs).shuffle(shuffle_buffer).batch(
            batch_size).prefetch(prefetch_buffer)
        client_datasets.append(dataset)

    federated_train_data = client_datasets

    def create_keras_model():
        if network == "LSRM":
            model = tf.keras.models.Sequential([
                tf.keras.layers.LSTM(64, input_shape=(seq_length, input_size)),
                tf.keras.layers.Dense(output_size, activation="linear")
            ])
        else:
            model = tf.keras.models.Sequential([
                tf.keras.layers.SimpleRNN(64, input_shape=(seq_length, input_size)),
                tf.keras.layers.Dense(output_size, activation="linear")
            ])
        return model

    def model_fn():
        keras_model = create_keras_model()
        return tff.learning.models.from_keras_model(
            keras_model,
            input_spec=federated_train_data[0].element_spec,
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanSquaredError()])

    training_process = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(client_learning_rate),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(server_learning_rate))

    train_state = training_process.initialize()

    for round_num in range(epochs):
        result = training_process.next(train_state, federated_train_data)
        train_state = result.state
        train_metrics = result.metrics
        print('round {:2d}, metrics={}'.format(round_num, train_metrics))


if __name__ == "__main__":
    train()
