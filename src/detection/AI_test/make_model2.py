# MobileNet
import os

import tensorflow as tf
from tensorflow import keras

# import numpy as np
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.applications.imagenet_utils import preprocess_input
# from tensorflow.keras.applications.imagenet_utils import decode_predictions
# from tensorflow.keras.preprocessing import image
# import keras



model = tf.keras.applications.mobilenet_v2.MobileNetV2(
    input_shape=(224, 224, 3),
    alpha=0.35,
    include_top=True,
    weights=None,
    input_tensor=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax',
    # **kwargs
)

print(model.summary())

model.save('newmodel_224.h5')