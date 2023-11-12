import tensorflow as tf
class MulticlassTruePositives(tf.keras.metrics.Metric):
    def __init__(self, name='multiclass_true_positives', num_classes=10, **kwargs):
        super(MulticlassTruePositives, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.confusion_matrix = self.add_weight(
            name='confusion_matrix',
            shape=(num_classes, num_classes),
            initializer='zeros',
            dtype=tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis=1)
        values = tf.cast(y_true, 'int32') == tf.cast(y_pred, 'int32')
        values = tf.cast(values, 'float32')

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, 'float32')
            values = tf.multiply(values, sample_weight)

        confusion_update = tf.math.confusion_matrix(
            y_true, y_pred, num_classes=self.num_classes, dtype=tf.float32
        )

        self.confusion_matrix.assign_add(confusion_update)

    def result(self):
        return self.confusion_matrix

    def reset_states(self):
        self.confusion_matrix.assign(tf.zeros_like(self.confusion_matrix))

    def get_config(self):
        config = super(MulticlassTruePositives, self).get_config()
        config['num_classes'] = self.num_classes
        return config


class TruePred(tf.keras.metrics.Metric):
    def __init__(self, name='get_true_pred', batch_size=64, output_size=96, **kwargs):
        super(TruePred, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', shape=(batch_size, output_size, 2), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = tf.cast(y_pred, 'float32'), tf.cast(y_true, 'float32')
        values = tf.stack(values, axis=-1)
        self.true_positives.assign_add(values)

    def result(self):
        return self.true_positives

    def reset_states(self):
        self.true_positives.assign(tf.zeros_like(self.true_positives))

