import tensorflow as tf
import tensorflow_federated as tff

""" 
Testing to see if tensorflow and federated are installed correctly
Prints out the available GPUs, if no GPUs are  available,
learning will be performed on the CPU
"""

print(tff.__version__)
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
