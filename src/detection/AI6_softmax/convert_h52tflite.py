import tensorflow as tf
from keras.layers import (
    Dense,
    Input,
    GlobalAveragePooling2D,
    Dropout,
    BatchNormalization,
    UpSampling2D,
    Conv2D,
    MaxPooling2D,
    AveragePooling2D,
    Flatten,
    Concatenate,
)
from tensorflow.keras.optimizers import Adam
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import CSVLogger
from keras.models import Model
from tensorflow.keras.applications import MobileNet, MobileNetV2
from tensorflow import keras
from keras.callbacks import ModelCheckpoint


# import tensorflow_model_optimization as tfmot
from keras import backend as K
from tqdm import tqdm
import os
import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import cv2
from module.const import *
from module import func, loss
import random
from keras.models import load_model
import keras.backend as K

custom_model = keras.models.load_model("model/base_model_96_cut.h5")

custom_model.summary()

# 層の数を取得
num_layers = len(custom_model.layers)
print("層の数:", num_layers)

# # 特定の名前の層のインデックスを取得
# target_layer_name = "block_6_expand_BN[0][0]"  # 変更が必要な層の名前に置き換える

# try:
#     target_layer_index = [layer.name for layer in custom_model.layers].index(target_layer_name)
#     print(f"{target_layer_name}の層のインデックス: {target_layer_index}")
# except ValueError:
#     print(f"{target_layer_name}の層は見つかりませんでした")

# custom_model = keras.models.load_model("model/custom_model_gloc.h5")

# custom_model.summary()

# # 層の数を取得
# num_layers = len(custom_model.layers)
# print("層の数:", num_layers)



# # custom_model.save(FULL_MODEL_PATH)
# conveter = tf.lite.TFLiteConverter.from_keras_model(custom_model)
# tflite_model = conveter.convert()
# float_model_size = len(tflite_model) / 1024 / 1024
# print(f"float model size: {float_model_size} MB")
# open("model/base_model_96_cut.tflite", "wb").write(tflite_model)