import cv2
import os
import random
import shutil
import FileUtils.img_name_helper as imh

Data_dir = '/home/shunya/Datasets/indoor-scenes-cvpr-2019/indoorCVPR_09/Images'
Train_dir = '/home/shunya/PythonProjects/NethraAngleRot/dataset/train'
Valid_dir = '/home/shunya/PythonProjects/NethraAngleRot/dataset/validation'

def create_train_dir():
    all_img_path_list = imh.get_files_list(Data_dir, 'jpg')
    img_path_list = []
    min_sz = 200
    for index, img_path in enumerate(all_img_path_list):
        img = cv2.imread(img_path)
        if img is None:
            continue
        ht, wd, _ = img.shape
        if min(ht, wd) > min_sz:
            img_dir, img_name = os.path.split(img_path)
            img_new_path = os.path.join(Train_dir, img_name)
            shutil.copy2(img_path, img_new_path)
            print(index)

def create_valid_dir():
    img_path_list = imh.get_files_list(Train_dir, 'jpg')
    random.shuffle(img_path_list)
    valid_index = int(0.2 * len(img_path_list))
    img_path_valid_list = img_path_list[0:valid_index]

    for index, img_path in enumerate(img_path_valid_list):
        img_dir, img_name = os.path.split(img_path)
        img_new_path = os.path.join(Valid_dir, img_name)
        shutil.move(img_path, img_new_path)
        print(index)

#create_valid_dir()
#create_train_dir()