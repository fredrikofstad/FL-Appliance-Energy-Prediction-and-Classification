import tensorflow as tf
import tensorflow_federated as tff
from preprocess import *
import data


def train(
        epochs=2,
        batch_size=96,
        shuffle_buffer=1,
        prefetch_buffer=1,
        seq_length=96,
        input_size=7,
        output_size=1
):
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

    def create_keras_model2():
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(64, input_shape=(seq_length, input_size)),
            tf.keras.layers.Dense(output_size, activation="linear")
        ])
        return model

    def model_fn():
        keras_model = create_keras_model2()
        return tff.learning.models.from_keras_model(
            keras_model,
            input_spec=federated_train_data[0].element_spec,
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanSquaredError()])

    training_process = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

    train_state = training_process.initialize()

    for round_num in range(epochs):
        result = training_process.next(train_state, federated_train_data)
        train_state = result.state
        train_metrics = result.metrics
        print('round {:2d}, metrics={}'.format(round_num, train_metrics))
