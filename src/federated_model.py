import tensorflow as tf
import tensorflow_federated as tff
from preprocess import *
import data

consumer_data = {}

dataframe, name_list = data.create_df()
for sheet_name, df in dataframe.items():
    consumer_data[sheet_name] = df

processed_data = [prepare_tensors(consumer_data[name_list[i]]) for i in range(1, 50)]

input_data_list = [processed_data[i][0].numpy() for i in range(49)]
target_data_list = [processed_data[i][1].numpy() for i in range(49)]


NUM_CLIENTS = len(input_data_list)
NUM_EPOCHS = 2
BATCH_SIZE = 96
SHUFFLE_BUFFER = 1
PREFETCH_BUFFER = 1
SEQ_LENGTH = 96
INPUT_SIZE = 7
OUTPUT_SIZE = 1  # Binary classification output

def create_tf_dataset(input_data, target_data):
    dataset = tf.data.Dataset.from_tensor_slices((input_data, target_data))
    return dataset

client_datasets = []
for i in range(NUM_CLIENTS):
    dataset = create_tf_dataset(input_data_list[i], target_data_list[i])
    dataset = dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(
        BATCH_SIZE).prefetch(PREFETCH_BUFFER)
    client_datasets.append(dataset)

federated_train_data = client_datasets

def create_keras_model2():
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, input_shape=(SEQ_LENGTH, INPUT_SIZE)),
        tf.keras.layers.Dense(OUTPUT_SIZE, activation="linear")
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

for round_num in range(NUM_EPOCHS):
    result = training_process.next(train_state, federated_train_data)
    train_state = result.state
    train_metrics = result.metrics
    print('round {:2d}, metrics={}'.format(round_num, train_metrics))