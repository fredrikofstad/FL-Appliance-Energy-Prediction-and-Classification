import tensorflow as tf
import tensorflow_federated as tff


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print(tff.__version__)
print(tf.__version__)
