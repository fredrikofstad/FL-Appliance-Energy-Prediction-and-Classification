from data import *


class Household:
    def __init__(self, index: int):
        self.index = index
        self.df = sum_energy_consumption(read_to_df(index))
        self.io_pairs = create_io_pairs(self.df)
        self.data_train, self.data_test, self.labels_train, self.labels_test = create_dataset(self.df)

