import os

import tensorflow as tf
import keras

# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model

# model = tf.keras.models.load_model('trained_model.h5')
model = tf.keras.models.load_model('random_model_96.h5')

layer_names = [l.name for l in model.layers]

print(layer_names)

print(layer_names.index('block_6_expand_relu')) #56

new_model = Model(inputs=model.layers[0].input, outputs=model.layers[56].output)

new_model.summary()

new_model.save('random_model_96_cut.h5')

# keras.utils.plot_model(new_model, "mini_resnet.png", show_shapes=True)

# モデルの構造を表示（長いよ）
# print(model.summary())
# print(new_model.summary())

