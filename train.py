
import numpy as np
import pandas as pd 
import cv2
import pickle
import csv
import tensorflow as tf
from PIL import Image
import os
import argparse
import keras
import sys

from keras.preprocessing import image
from tensorflow.keras.preprocessing import image as im
import time
import datetime

from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from keras import regularizers
from keras import backend as K
from sklearn.preprocessing import OneHotEncoder

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, Model
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Input, AveragePooling2D
from keras.layers import Activation, Flatten, Dense, Dropout

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default='2',
        type=str,
        help="num of device"
    )
    
    parser.add_argument(
        "--train-dir",
        default='/home/z_andrei/datasets/Traffic_ligths_data/TL_ALL/train/',
        type=str,
        help="train dir"
    )
    parser.add_argument(
        "--val-dir",
        default='/home/z_andrei/datasets/Traffic_ligths_data/TL_ALL/test/',
        type=str,
        help="val dir"
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


def main(args):
    tf.keras.backend.set_learning_phase(1) 
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    now = datetime.datetime.now()
    timeset = now.strftime("%d-%m-%Y_%H:%M")


    # 1) Path to train and test datasets
    train_dir = args.train_dir  
    test_dir = args.val_dir


    # 2) Target width and height of input images, number of classes, number of train and test images, batch size, epochs
    img_height, img_width = 96, 32
    n_classes = len(os.listdir(train_dir))
    nb_train_samples = sum([len(os.listdir(train_dir +'/'+ folder)) for folder in os.listdir(train_dir)])
    nb_test_samples = sum([len(os.listdir(test_dir +'/'+ folder)) for folder in os.listdir(test_dir)])
    batch_size = 128
    epochs = 100

    # 4) Compile parametrs
    loss_ = 'categorical_crossentropy'
    optimizer_ = 'adam'
    metrics_ = ['acc']
    
    train_datagen = ImageDataGenerator(
    rescale = 1.0 / 255,
    zoom_range = 0.2,
    rotation_range = 5,
    width_shift_range = img_width // 3,
    height_shift_range = img_height // 3,
    horizontal_flip = True,
    vertical_flip = False)
    test_datagen = ImageDataGenerator(rescale = 1.0 / 255)
    
    train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode = 'categorical')
    
    test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode = 'categorical', shuffle=False)
    
    model = resNetM((img_height, img_width, 3), n_classes)
    model.compile(loss = loss_, optimizer = optimizer_, metrics = metrics_)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=8, verbose=1, min_delta=1e-4) # остановка обучения, если loss на валидационном множесте улучшается менее чем на 10^-4
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, min_delta=1e-4)
    filepath="models/resnetm-all-71-{epoch:02d}-{val_acc:.2f}.hdf5"
    check = ModelCheckpoint(filepath, monitor = "val_acc", save_best_only = False) # сохранение лучшей (с наибольшим acc на валидационном множестве) сети
    callbacks_list = [early_stop, reduce_lr , check]
    
    model_history = model.fit_generator(
    train_generator,
    epochs = epochs,
    validation_data = test_generator,
    validation_steps = nb_test_samples // batch_size,
    callbacks = callbacks_list,
    steps_per_epoch = nb_train_samples // batch_size)
    
if __name__=="__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
