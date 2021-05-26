import os
import math
import tensorflow as tf

from src_train_model.img_classification_csv_input import ImageClassificationCSVDataPipeline
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from config.img_classification_config import ConfigObj
from models.create_model import get_custom_model
import math
import utils

from src_optimal_lr_finder.LRFinder import LRFinder

## TF dataset
from FileUtils.img_name_helper import get_files_list
Train_dir = '/home/shunya/PythonProjects/NethraAngleRot/dataset/train'
train_file_names = get_files_list(Train_dir, 'jpg')

Valid_dir = '/home/shunya/PythonProjects/NethraAngleRot/dataset/validation'
validation_file_names = get_files_list(Valid_dir, 'jpg')

## Get model
custom_model = get_custom_model()

## Prepare for training
batch_sz = ConfigObj.batch_size
input_shape = (ConfigObj.img_dim, ConfigObj.img_dim, ConfigObj.img_channels)
num_epochs = ConfigObj.epochs

train_steps_per_epoch = math.ceil(len(train_file_names) / batch_sz)
val_steps_per_epoch = math.ceil(len(validation_file_names) / batch_sz)

## LR Finder
min_lr, max_lr = 0.00001, 0.1
lr_finder = LRFinder(min_lr, max_lr)
lr_log_file_path = os.path.join(ConfigObj.Path_Parent_Dir, 'src_optimal_lr_finder', 'optimal_lr.log')

## Train Start:

history_freeze = custom_model.fit(
    utils.RotNetDataGenerator(
        train_file_names,
        input_shape=input_shape,
        batch_size=batch_sz,
        preprocess_func=preprocess_input,
        crop_center=True,
        crop_largest_rect=True,
        shuffle=True
    ),
    steps_per_epoch=train_steps_per_epoch, epochs=num_epochs,
    validation_data=utils.RotNetDataGenerator(
        validation_file_names,
        input_shape=input_shape,
        batch_size=batch_sz,
        preprocess_func=preprocess_input,
        crop_center=True,
        crop_largest_rect=True
    ),
    validation_steps=val_steps_per_epoch, callbacks=[lr_finder])

lr_finder.dump(lr_log_file_path)
print('----> LR EXPERIMENT FINISHED <----')