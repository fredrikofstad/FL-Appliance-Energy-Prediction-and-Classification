import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from data import *
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical


class Household:
    def __init__(self, index: int):
        self.model = None
        self.index = index
        self.df = sum_energy_consumption(read_to_df(index))
        self.io_pairs = create_io_pairs(self.df)
        self.data_train, self.data_test, self.labels_train, self.labels_test = create_dataset(self.df)
        self.class_num = len(get_appliance_names())
        self.y_train_encoded, self.y_test_encoded = self._encode(self.labels_train, self.labels_test)

    def _encode(self, y_train, y_test):
        y_train_mapped = [appliance_mapping[label] for label in y_train]
        y_test_mapped = [appliance_mapping[label] for label in y_test]
        y_train_categorical = to_categorical(y_train_mapped, num_classes=self.class_num)
        y_test_categorical = to_categorical(y_test_mapped, num_classes=self.class_num)
        return y_train_categorical, y_test_categorical

    def build_classifier(self):
        # build the model
        self.model = Sequential()
        self.model.add(Dense(units=64, activation='relu', input_dim=1))
        self.model.add(Dense(units=self.class_num, activation='softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def train(self, epochs=10, batch_size=1):
        X_train = np.array(self.data_train).reshape(-1, 1)
        self.model.fit(
            X_train,
            self.y_train_encoded,
            epochs=epochs,
            batch_size=batch_size)

    def evaluate(self):
        X_test = np.array(self.data_test).reshape(-1, 1)
        return self.model.evaluate(X_test, self.y_test_encoded)

