import cv2
import os
import random
import shutil
import FileUtils.img_name_helper as imh
from utils import generate_rotated_image
import numpy as np
import csv
import FileUtils.img_name_helper as imh

Data_dir = '/home/shunya/Datasets/indoor-scenes-cvpr-2019/indoorCVPR_09/Images'
Train_dir = '/home/shunya/PythonProjects/NethraAngleRot/dataset/train'
Valid_dir = '/home/shunya/PythonProjects/NethraAngleRot/dataset/validation'
Test_dir = '/home/shunya/PythonProjects/NethraAngleRot/dataset/test'
Test_gt_csv = '/home/shunya/PythonProjects/NethraAngleRot/dataset/test/angle_gt.csv'

input_shape = (224, 224, 3)

def creat_test_dataset():
    crop_center = True
    crop_largest_rect = True
    img_path_list = imh.get_files_list(Train_dir, 'jpg')
    random.shuffle(img_path_list)
    f = open(Test_gt_csv, 'w')
    writer = csv.writer(f, delimiter=',')
    writer.writerow(['Name', 'Angle'])

    for ii in range(360):
        img = cv2.imread(img_path_list[ii])
        rotation_angle = np.random.randint(360)

        rotated_image = generate_rotated_image(
            img,
            rotation_angle,
            size=input_shape[:2],
            crop_center=crop_center,
            crop_largest_rect=crop_largest_rect
        )
        img_path, img_name = imh.get_path_name_frm_index(ii, Test_dir)
        cv2.imwrite(img_path, rotated_image)
        lst_gt = [img_name, rotation_angle]
        writer.writerow(lst_gt)
        print(ii)

    f.close()


def one_sample():
    crop_center = True
    crop_largest_rect = True
    img_path_list = imh.get_files_list(Train_dir, 'jpg')
    random.shuffle(img_path_list)

    img = cv2.imread(img_path_list[0])
    rotation_angle = np.random.randint(360)

    rotated_image = generate_rotated_image(
        img,
        rotation_angle,
        size=input_shape[:2],
        crop_center=crop_center,
        crop_largest_rect=crop_largest_rect
    )

    cv2.imwrite('img1.png', img)
    cv2.imwrite('img2.png', rotated_image)
    print(rotation_angle)


one_sample()
#creat_test_dataset()



