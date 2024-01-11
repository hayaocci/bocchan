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

# # モデルを読み込む
# model = MobileNetV2(weights='imagenet')
# print("\n◆Model:")
# print(f"{model.name}")

# model.export(export_dir='model', export_format=ExportFormat.LABEL)

# model.save('base_model.h5')
# model.save('saved_model')

new_model = tf.keras.models.load_model('trained_model.h5')

# モデルの構造を表示（長いよ）
# print(model.summary())
print(new_model.summary())