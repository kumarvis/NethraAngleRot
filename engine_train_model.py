import tensorflow as tf
from src_train_model.img_classification_csv_input import ImageClassificationCSVDataPipeline
from tensorflow.keras.applications.imagenet_utils import preprocess_input
from config.img_classification_config import ConfigObj
from models.create_model import get_custom_model
import math
import utils

## TF dataset
from FileUtils.img_name_helper import get_files_list
Train_dir = '/home/shunya/PythonProjects/NethraAngleRot/dataset/train'
train_file_names = get_files_list(Train_dir, 'jpg')

Valid_dir = '/home/shunya/PythonProjects/NethraAngleRot/dataset/validation'
validation_file_names = get_files_list(Train_dir, 'jpg')

## Get model
custom_model = get_custom_model()

## Prepare for training
batch_sz = ConfigObj.batch_size
input_shape = (ConfigObj.img_dim, ConfigObj.img_dim, ConfigObj.img_channels)
num_epochs = ConfigObj.epochs

train_steps_per_epoch = math.ceil(len(train_file_names) / batch_sz)
val_steps_per_epoch = math.ceil(len(validation_file_names) / batch_sz)

## callbacks
from callbacks.custom_callbacks import MyCallBacks

my_call_backs_obj = MyCallBacks(train_steps_per_epoch)
lst_my_callbacks = my_call_backs_obj.get_list_callbacks()

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
    validation_steps=val_steps_per_epoch, callbacks=lst_my_callbacks)

## Plottings
from src_train_model.plot_keras_hist import dump_hist_data, plot_hist_frm_csv
hist_csv_path = dump_hist_data(history_freeze, prefix='base')
plot_hist_frm_csv(hist_csv_path, prefix='base')

print('----> EXPERIMENT FINISHED <----')


