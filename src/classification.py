import importlib
import tensorflow as tf
import tensorflow_federated as tff
from preprocess import prepare_classification_tensors
from metrics import MulticlassTruePositives
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

    # Create a tf.data.Dataset from the loaded data
    def create_tf_dataset(input_data, target_data):
        return tf.data.Dataset.from_tensor_slices((input_data, target_data))

    consumer_data = {}

    dataframe, name_list = data.create_df()
    for sheet_name, df in dataframe.items():
        consumer_data[sheet_name] = df

    processed_data = [prepare_classification_tensors(consumer_data[name_list[i]]) for i in range(0, 50)]

    input_data_train = [processed_data[i][0].numpy() for i in range(50)]
    target_data_train = [processed_data[i][1].numpy() for i in range(50)]
    input_data_list_test = [processed_data[i][2].numpy() for i in range(50)]
    target_data_list_test = [processed_data[i][3].numpy() for i in range(50)]

    num_clients = len(input_data_train)

    client_datasets = []
    for i in range(num_clients):
        dataset = create_tf_dataset(input_data_train[i], target_data_train[i])
        dataset = dataset.repeat(epochs).shuffle(shuffle_buffer).batch(
            batch_size).prefetch(prefetch_buffer)
        client_datasets.append(dataset)

    client_datasets_test = []
    for i in range(num_clients):
        dataset = create_tf_dataset(input_data_list_test[i], target_data_list_test[i])
        dataset = dataset.repeat(epochs).shuffle(shuffle_buffer).batch(
            batch_size).prefetch(prefetch_buffer)
        client_datasets.append(dataset)

    federated_train_data = client_datasets

    def create_keras_model():
        if network == "LSRM":
            model = tf.keras.models.Sequential([
                tf.keras.layers.LSTM(64, input_shape=(input_size, seq_length)),
                tf.keras.layers.Dense(output_size, activation="sigmoid")
            ])
        else:
            model = tf.keras.models.Sequential([
                tf.keras.layers.SimpleRNN(64, input_shape=(input_size, seq_length)),
                tf.keras.layers.Dense(output_size, activation="sigmoid")
            ])
        return model

    def create_keras_metric():
        return MulticlassTruePositives(num_classes=output_size)

    def model_fn():
        keras_model = create_keras_model()
        return tff.learning.models.from_keras_model(
            keras_model,
            input_spec=federated_train_data[0].element_spec,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(), create_keras_metric()])

    training_process = tff.learning.algorithms.build_weighted_fed_avg(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(client_learning_rate),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(server_learning_rate))

    train_state = training_process.initialize()

    confusion_list = []
    accuracy_list = []
    loss_values = []

    for round_num in range(epochs):
        result = training_process.next(train_state, federated_train_data)
        train_state = result.state
        train_metrics = result.metrics
        confusion_list.append(train_metrics['client_work']["train"]['multiclass_true_positives'])
        accuracy_list.append(train_metrics['client_work']["train"]['sparse_categorical_accuracy'])
        loss_values.append(train_metrics['client_work']['train']['loss'])
        print(f"Metrics for round {round_num}: {train_metrics}")

    #Evaluation setup
    evaluation_process = tff.learning.algorithms.build_fed_eval(model_fn)

    evaluation_state = evaluation_process.initialize()
    model_weights = training_process.get_model_weights(train_state)
    evaluation_state = evaluation_process.set_model_weights(evaluation_state, model_weights)

    evaluation_output = evaluation_process.next(evaluation_state, client_datasets_test)
    evaluation_metrics = evaluation_output[1]['client_work']['eval']['total_rounds_metrics']
    multiclass_true_positives = evaluation_metrics['multiclass_true_positives']

    return loss_values, confusion_list, accuracy_list, multiclass_true_positives


if __name__ == "__main__":
    train()
