
import numpy as np
import pandas as pd 
import cv2
import pickle
import csv
import tensorflow as tf
import random
from PIL import Image
import os
import argparse
import keras
import sys

from keras.preprocessing import image
from tensorflow.keras.preprocessing import image as im
import time
from shutil import copyfile
import datetime

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from keras import regularizers
from keras import backend as K
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, Model
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Input, AveragePooling2D
from keras.layers import Activation, Flatten, Dense, Dropout

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default='0',
        type=str,
        help="num of device"
    )
    parser.add_argument(
        "--test-dir",
        default='/home/z_andrei/datasets/Traffic_ligths_data/TL_ALL/test/',
        type=str,
        help="val dir"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="model path"
    )
    parser.add_argument(
        "--error-logs",
        type=int,
        default=0,
        help="model path"
    )
    return parser


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x

def resNetM(input_shape, classes):
    img_input = Input(shape=input_shape)

    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', name='conv1')(img_input)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = AveragePooling2D((3, 3), name='avg_pool')(x)

    x = Flatten()(x)
    x = Dense(classes, activation='softmax', name='fc5')(x)

    inputs = img_input

    model = Model(inputs, x, name='resnetM')

    return model

def create_subfolders(folder, subfolders, s_time):
    os.mkdir(folder+'/'+str(s_time))
    os.mkdir(folder+'/'+str(s_time)+'/false_positive')
    os.mkdir(folder+'/'+str(s_time)+'/false_negative')
    FP = folder+'/'+str(s_time)+'/false_positive'+'/'
    FN = folder+'/'+str(s_time)+'/false_negative'+'/'
    for subfolder in subfolders:
        os.mkdir(FP+'/'+subfolder)
        os.mkdir(FN+'/'+subfolder)
        
def save_error_matrix(test_path, model, labels, subfolders, cls_idx, idx_cls):
    if not os.path.exists("error_logs"):
        os.mkdir("error_logs")
    s_time = "".join("_".join(time.ctime().split()).split(":"))
        
    create_subfolders("error_logs", subfolders, s_time)
    FP = "error_logs"+'/'+str(s_time)+'/false_positive'+'/'
    FN = "error_logs"+'/'+str(s_time)+'/false_negative'+'/'
    
    for folder in os.listdir(test_path):
        true = cls_idx[folder]
        for im in os.listdir(test_path+'/'+folder):
            image_name = test_path+'/'+folder+'/'+im
            x = image.load_img(image_name, target_size=(96, 32))
            x = np.array(x, dtype=float)
            x = x/255.0
            x = np.expand_dims(x, axis=0)
            pred = model.predict(x)
            pred = np.argmax(pred, axis=1)[0]
            if not pred == true:
                copyfile(image_name, FP+idx_cls[pred]+'/'+im)
                copyfile(image_name, FN+idx_cls[true]+'/'+im)


def main(args):
    tf.keras.backend.set_learning_phase(0)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    now = datetime.datetime.now()
    timeset = now.strftime("%d-%m-%Y_%H:%M")
    img_height, img_width = 96, 32
    
    model_name = args.model
    loaded_model = load_model(model_name)
    
    # Speed test using generator

    test_dir = args.test_dir
    
    test_datagen = ImageDataGenerator(rescale = 1.0 / 255)

    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size = (img_height, img_width),
        batch_size = 1,
        class_mode = 'categorical', shuffle=False)

    nb_samples = len(test_generator.filenames)
    cls_idx = test_generator.class_indices
    idx_cls = {val:key for key, val in cls_idx.items()}
    
    
    labels = test_generator.classes
    if args.error_logs:
        save_error_matrix(test_dir, loaded_model, labels, os.listdir(test_dir), cls_idx, idx_cls)

    start_time = time.time()
    predictions = loaded_model.predict_generator(test_generator, verbose=1)#, steps = nb_samples)
    print("Average time to pridict one photo using generator is", nb_samples/(time.time() - start_time))
    predictions = np.array(predictions)
    predictions = np.argmax(predictions, axis=1)
    
    from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
    
    precision, recall, fscore, support = precision_recall_fscore_support(predictions, labels, labels=np.unique(labels))
    
    print(classification_report(labels, predictions))
    
if __name__=="__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
