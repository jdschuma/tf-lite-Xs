from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf

assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging

logging.set_verbosity(logging.ERROR)

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))


def train_variation(dirname, _model_name, epoch_count):
    filename_train = dirname + r'train/x.tfrecord'
    filename_eval = dirname + r'valid/x.tfrecord'
    dataset_training = object_detector.DataLoader(tfrecord_file_patten=filename_train, size=1579, label_map=["x"])
    dataset_evaluation = object_detector.DataLoader(tfrecord_file_patten=filename_eval, size=241, label_map=["x"])
    spec = model_spec.get(_model_name)
    model = object_detector.create(dataset_training, model_spec=spec, batch_size=32, train_whole_model=True,
                                   epochs=epoch_count,
                                   validation_data=dataset_evaluation)
    model.evaluate(dataset_evaluation)

    model.export(export_dir=dirname, tflite_filename='Xs-' + str(epoch_count) + '-' + _model_name + '.tflite')


train_variation(r'dataset/Xs.v1i.tfrecord/', 'efficientdet_lite0', 10)
train_variation(r'dataset/Xs.v1i.tfrecord/', 'efficientdet_lite0', 25)
train_variation(r'dataset/Xs.v1i.tfrecord/', 'efficientdet_lite0', 50)
train_variation(r'dataset/Xs.v1i.tfrecord/', 'efficientdet_lite0', 100)

train_variation(r'dataset/Xs.v1i.tfrecord/', 'efficientdet_lite1', 10)
train_variation(r'dataset/Xs.v1i.tfrecord/', 'efficientdet_lite1', 25)
train_variation(r'dataset/Xs.v1i.tfrecord/', 'efficientdet_lite1', 50)
train_variation(r'dataset/Xs.v1i.tfrecord/', 'efficientdet_lite1', 100)

train_variation(r'dataset/Xs.v1i.tfrecord/', 'efficientdet_lite2', 10)
train_variation(r'dataset/Xs.v1i.tfrecord/', 'efficientdet_lite2', 25)
train_variation(r'dataset/Xs.v1i.tfrecord/', 'efficientdet_lite2', 50)
train_variation(r'dataset/Xs.v1i.tfrecord/', 'efficientdet_lite2', 100)
