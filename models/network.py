import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Flatten
from config.img_classification_config import ConfigObj

def create_network():
    input_shape = (ConfigObj.img_dim, ConfigObj.img_dim, ConfigObj.img_channels)
    num_classes = ConfigObj.Num_Classes

    if ConfigObj.Network_Architecture == 'ResNet50':
        base_model = tf.keras.applications.resnet50.ResNet50(
        include_top=False, weights='imagenet', input_shape=input_shape)

    elif ConfigObj.Network_Architecture == 'EfficientnetB3':
        base_model = tf.keras.applications.efficientnet.EfficientNetB3(
            include_top=False, weights='imagenet', input_shape=input_shape)

    elif ConfigObj.Network_Architecture == 'MobileNetV2':
        base_model = tf.keras.applications.MobileNetV2(
            include_top=False, weights='imagenet', input_shape=input_shape)

    #for layer in base_model.layers:
        #layer.trainable = False
    #for layer in base_model.layers[-26:]:
        #layer.trainable = True

    x = base_model.output
    x = Flatten()(x)
    final_output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=final_output)

    return model