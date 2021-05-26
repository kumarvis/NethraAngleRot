import tensorflow as tf
from tensorflow import  keras
from src_train_model.img_classification_csv_input import ImageClassificationCSVDataPipeline
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from config.img_classification_config import ConfigObj
from models.create_model import get_custom_model
import math
import utils
import os
import csv
import cv2
import numpy as np

## TF dataset
from FileUtils.img_name_helper import get_files_list
Test_dir = '/home/shunya/PythonProjects/NethraAngleRot/dataset/test'
gt_csv_name = 'angle_gt.csv'
csv_gt_name = os.path.join(Test_dir, gt_csv_name)

model_path = '/home/shunya/PythonProjects/NethraAngleRot/checkpoints/weights-epoch200-loss1.64.h5'
model = keras.models.load_model(model_path)

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

with open(csv_gt_name) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    next(csv_reader)
    max_diff = -1
    error_sum = 0.0
    count = 0
    for row in csv_reader:
        img_name, angle = row[0], int(row[1])
        img_path = os.path.join(Test_dir, img_name)
        img = get_image(img_path)
        x = np.expand_dims(img, axis=0)
        x = preprocess_input(x)
        #ans = model(x, training=False)
        pred_angle = tf.math.argmax(model(x, training=False), axis=1).numpy()
        pred_angle = pred_angle[0]
        abs_diff = abs(pred_angle - angle)
        count = count + 1
        if abs_diff > 10:
            print('Image Name = ', img_name, abs_diff)
        error_sum = error_sum + abs_diff
        if abs_diff > max_diff:
            max_diff = abs_diff

    print('Avg Error = ', error_sum / 360.0)
    print('Max Error = ', max_diff)









print('----> EXPERIMENT FINISHED <----')


