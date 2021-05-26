import tensorflow as tf
from tensorflow import  keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input
import math

import os
import csv
import cv2
import numpy as np
import sys


def get_image(img_path):
    image = cv2.imread(img_path)
    input_size = (224, 224)
    image = cv2.resize(image, input_size)

    start_points = [(54, 0), (108, 0), (162, 0), (0, 54), (0, 108), (0, 162)]
    end_points = [(54, 223), (108, 223), (162, 223), (223, 54), (223, 108), (223, 162)]
    color = (0, 0, 0)
    thickness = 1

    for index, item in enumerate(start_points):
        pt1 = start_points[index]
        pt2 = end_points[index]
        image = cv2.line(image, pt1, pt2, color, thickness)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

def get_label(img):
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    pred_angle = tf.math.argmax(model(x, training=False), axis=1).numpy()
    pred_angle = pred_angle[0]
    return pred_angle

num = len(sys.argv)
if num < 3:
    print('Invalid number of arguments')
    exit(-1)

model_path = sys.argv[1]
model = keras.models.load_model(model_path)

if num > 3:
    img_path1 = sys.argv[2]
    img_path2 = sys.argv[3]

    img1 = get_image(img_path1)
    img2 = get_image(img_path2)

    theta1 = get_label(img1)
    theta2 = get_label(img2)
    print('Angle Between Images = ', abs(theta1-theta2))

elif num == 3:
    img_path1 = sys.argv[2]
    img1 = get_image(img_path1)
    theta1 = get_label(img1)
    print('Angle Between Images = ', abs(theta1))





