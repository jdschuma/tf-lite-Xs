from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

print("TensorFlow version: {}".format(tf.__version__))
print("Eager execution: {}".format(tf.executing_eagerly()))

dirname= r'dataset/xs.v2-xs-nonoise.tfrecord/'

filenameTrain = dirname + r'train/xs.tfrecord'
filenameEval = dirname + r'valid/xs.tfrecord'
filenameTest = dirname + r'test/xs.tfrecord'

datasetTraining = object_detector.DataLoader(tfrecord_file_patten=filenameTrain, size=285, label_map=["x"])
datasetEvaluation = object_detector.DataLoader(tfrecord_file_patten=filenameEval, size=26, label_map=["x"])
datasetTesting = object_detector.DataLoader(tfrecord_file_patten=filenameTest, size=14, label_map=["x"])

spec = model_spec.get('efficientdet_lite0')
model = object_detector.create(datasetTraining, model_spec=spec, batch_size=16, train_whole_model=True, epochs=50, validation_data=datasetEvaluation )
model.evaluate(datasetTesting)
model.export(export_dir=dirname, tflite_filename='Xs-50-lite0.tflite')


dirname= r'dataset/xs.v3-xs-lesschanges.tfrecord/'

filenameTrain = dirname + r'train/xs.tfrecord'
filenameEval = dirname + r'valid/xs.tfrecord'
filenameTest = dirname + r'test/xs.tfrecord'

datasetTraining = object_detector.DataLoader(tfrecord_file_patten=filenameTrain, size=285, label_map=["x"])
datasetEvaluation = object_detector.DataLoader(tfrecord_file_patten=filenameEval, size=26, label_map=["x"])
datasetTesting = object_detector.DataLoader(tfrecord_file_patten=filenameTest, size=14, label_map=["x"])

spec = model_spec.get('efficientdet_lite0')
model = object_detector.create(datasetTraining, model_spec=spec, batch_size=16, train_whole_model=True, epochs=50, validation_data=datasetEvaluation )
model.evaluate(datasetTesting)
model.export(export_dir=dirname, tflite_filename='Xs-50-lite0.tflite')

dirname= r'dataset/xs.v2-xs-nonoise.tfrecord/'

filenameTrain = dirname + r'train/xs.tfrecord'
filenameEval = dirname + r'valid/xs.tfrecord'
filenameTest = dirname + r'test/xs.tfrecord'

datasetTraining = object_detector.DataLoader(tfrecord_file_patten=filenameTrain, size=285, label_map=["x"])
datasetEvaluation = object_detector.DataLoader(tfrecord_file_patten=filenameEval, size=26, label_map=["x"])
datasetTesting = object_detector.DataLoader(tfrecord_file_patten=filenameTest, size=14, label_map=["x"])

spec = model_spec.get('efficientdet_lite1')
model = object_detector.create(datasetTraining, model_spec=spec, batch_size=16, train_whole_model=True, epochs=50, validation_data=datasetEvaluation )
model.evaluate(datasetTesting)
model.export(export_dir=dirname, tflite_filename='Xs-50-lite1.tflite')


dirname= r'dataset/xs.v3-xs-lesschanges.tfrecord/'

filenameTrain = dirname + r'train/xs.tfrecord'
filenameEval = dirname + r'valid/xs.tfrecord'
filenameTest = dirname + r'test/xs.tfrecord'

datasetTraining = object_detector.DataLoader(tfrecord_file_patten=filenameTrain, size=285, label_map=["x"])
datasetEvaluation = object_detector.DataLoader(tfrecord_file_patten=filenameEval, size=26, label_map=["x"])
datasetTesting = object_detector.DataLoader(tfrecord_file_patten=filenameTest, size=14, label_map=["x"])

spec = model_spec.get('efficientdet_lite1')
model = object_detector.create(datasetTraining, model_spec=spec, batch_size=16, train_whole_model=True, epochs=50, validation_data=datasetEvaluation )
model.evaluate(datasetTesting)
model.export(export_dir=dirname, tflite_filename='Xs-50-lite1.tflite')