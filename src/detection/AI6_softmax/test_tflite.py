from typing import Any
import tensorflow as tf
import os
import h5py
import numpy as np
from matplotlib import pyplot as plt
import cv2
import tensorflow.keras.backend as K
from module.const import *
from module import func, loss
import shutil
from tqdm import tqdm
import matplotlib.ticker as ticker
import csv

TEST_DIR = "taguchi_dataset/test/images"
SAVE_DIR = "test_result_main/test_quant2"
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR)


class TFLitePredictor:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, input_data):
        # input_data = input_data.astype(np.int8)
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]["index"])
        return output_data


# print('Quantized model accuracy: ',evaluate_model(interpreter_quant))
interpreter = TFLitePredictor(TFLITE_QUANT_MODEL_PATH)

# imgs = os.listdir(TEST_DIR)
# # imgs = imgs[::-1]
# count = 0
# for i, filename in enumerate(tqdm(imgs)):
#     img = cv2.imread(os.path.join(TEST_DIR, filename))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # img = np.expand_dims(img, axis=-1)

#     # img = np.array(img).astype
#     input_img = img.copy()
#     input_img -= 128
#     input_img = np.expand_dims(input_img, axis=0)
#     # print(input_img.shape)
#     input_img = input_img.astype(np.int8)
#     # print(input_img.dtype)
#     pred = interpreter(input_img)
#     plt.subplot(1, 3, 1)
#     # img += 128
#     plt.imshow(img)
#     plt.subplot(1, 3, 2)
#     plt.imshow(func.draw_line(input_img[0]))

#     plt.subplot(1, 3, 3)
#     plt.imshow(pred[0, :, :, 1], vmin=0, vmax=1)
#     save_path = os.path.join(SAVE_DIR, f"{i}.png")
#     plt.savefig(save_path)
#     plt.close()

test_img_dir = "taguchi_dataset/test/images"
test_label_dir = "taguchi_dataset/test/labels"
iou_list = []

for i, (img, label) in enumerate(zip(tqdm(os.listdir(test_img_dir)), os.listdir(test_label_dir))):
    img = cv2.imread(os.path.join(test_img_dir, img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = np.expand_dims(img, axis=-1)
    label = cv2.imread(os.path.join(test_label_dir, label))
    # print(type(img), type(label))
    # img = np.array(img).astype
    input_img = img.copy()
    input_img -= 128
    input_img = np.expand_dims(input_img, axis=0)
    # print(input_img.shape)
    input_img = input_img.astype(np.int8)
    # print(input_img.dtype)
    pred = interpreter(input_img)

    # print(label.shape, pred.shape)
    pred_flat = pred.reshape(-1, 2)
    label_flat = label.reshape(-1, 3)

    # Get binary masks for predictions and labels
    pred_mask = (pred_flat[:, 1] > 0.5).astype(int)  # Assuming class 1 is the positive class
    label_mask = (label_flat[:, 2] > 0.5).astype(int)  # Assuming class 2 is the positive class

    # Calculate intersection and union
    intersection = np.sum(pred_mask * label_mask)
    union = np.sum((pred_mask + label_mask) > 0)

    # Calculate IOU
    iou = intersection / union if union > 0 else 0.0
    iou_list.append(iou)

    

    # plt.subplots_adjust(wspace=0.2, top=0.1)
    plt.subplot(1, 3, 1)
    # img += 128
    plt.imshow(img, extent=[0, 96, 0, 96])
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    # plt.subplots_adjust(wspace=0.4)
    plt.subplot(1, 3, 2)
    # print(label.dtype)
    plt.imshow(label, extent=[0, 12, 0, 12])
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    plt.subplot(1, 3, 3)
    plt.imshow(pred[0, :, :, 1], vmin=0, vmax=1, extent=[0, 12, 0, 12])
    plt.locator_params(axis='x', nbins=4)
    plt.locator_params(axis='y', nbins=4)
    # plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9, wspace=0.4)
    plt.subplots_adjust(wspace=0.4)
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, f"{i}.png")
    plt.savefig(save_path)
    plt.close()

# write iou to csv
with open(os.path.join(SAVE_DIR, "iou.csv"), "w") as f:
    writer = csv.writer(f)
    writer.writerow(iou_list)

    # splited_img_lst = func.split_img(img)
    # for splited_img in splited_img_lst:
    #     splited_img = cv2.resize(splited_img, INPUT_SIZE)
    #     splited_img = np.array([splited_img])
    #     # print(splited_img.shape)
    #     splited_img = np.expand_dims(splited_img, axis=-1)
    #     splited_img -= 128

    #     pred = interpreter(splited_img)
    #     print(np.max(pred), np.min(pred))
    #     # pred += 128
    #     # pred /= 255
    #     splited_img = func.draw_line(splited_img[0])
    #     splited_img = (splited_img * 255).astype(np.uint8)
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(splited_img)
    #     plt.subplot(1, 2, 2)
    #     print(np.max(pred[0, :, :, 1]), np.min(pred[0, :, :, 1]))
    #     plt.imshow(pred[0, :, :, 1], vmin=0, vmax=1)
    #     save_path = os.path.join(SAVE_DIR, f"{count}.png")
    #     plt.savefig(save_path)
    #     plt.close()
    # count += 1

    # img = cv2.resize(img, INPUT_SIZE)
    # img_for_plot = img
    # img = np.array([img]) / 255
    # pred = interpreter(img)
    # img = func.draw_line(img[0])
    # img = (img * 255).astype(np.uint8)
    # plt.subplot(1, 2, 1)
    # plt.imshow(img)
    # plt.subplot(1, 2, 2)
    # plt.imshow(pred[0])

    # save_path = os.path.join(SAVE_DIR, f"{i}.png")
    # plt.savefig(save_path)
    # # plt.show()
    # plt.close()
