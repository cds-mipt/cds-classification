
import numpy as np
import pandas as pd 
import cv2
import pickle
import csv
import tensorflow
from PIL import Image
import os
import argparse
import tensorflow.keras
import sys

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import image as im
import time
import datetime

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback, TensorBoard
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from sklearn.preprocessing import OneHotEncoder

from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Input, AveragePooling2D
from tensorflow.keras.layers import Activation, Flatten, Dense, Dropout
from model import identity_block, conv_block, resNetM

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default='0',
        type=str,
        help="num of device"
    )
    
    parser.add_argument(
        "--train-dir",
        default='/home/cds-y/Datasets/synthetic-traffic-signs/signs_generated_100_train/',
        type=str,
        help="train dir"
    )
    parser.add_argument(
        "--val-dir",
        default='/home/cds-y/Datasets/synthetic-traffic-signs/signs_generated_100_tеst/',
        type=str,
        help="val dir"
    )
    parser.add_argument(
        "--callback-folder",
        default='./models/',
        type=str,
        help="callback folder"
    )
    return parser



def main(args):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    tensorflow.keras.backend.set_learning_phase(1)

    # 1) Path to train and test datasets
    train_dir = args.train_dir  
    test_dir = args.val_dir
    callback_folder = args.callback_folder


    # 2) Target width and height of input images, number of classes, number of train and test images, batch size, epochs
    #img_height, img_width = 96, 32 # for traffic light parameter
    img_height, img_width = 72, 72 # for traffic sign classification
    n_classes = len(os.listdir(train_dir))
    nb_train_samples = sum([len(os.listdir(train_dir +'/'+ folder)) for folder in os.listdir(train_dir)])
    nb_test_samples = sum([len(os.listdir(test_dir +'/'+ folder)) for folder in os.listdir(test_dir)])
    batch_size = 128
    epochs = 100

    # 4) Compile parametrs
    loss_ = 'categorical_crossentropy'
    #optimizer_ = 'adam'
    optimizer_ = tensorflow.keras.optimizers.Adam(
        learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False,
        name='Adam')

    metrics_ = ['acc']
    
    train_datagen = ImageDataGenerator(
    rescale = 1.0 / 255,
    zoom_range = 0.2,
    rotation_range = 5,
    width_shift_range = img_width // 3,
    height_shift_range = img_height // 3,
    horizontal_flip = False,
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

    timestr = time.strftime("%Y-%m-%d-%H-%M-%S")

    early_stop = EarlyStopping(monitor='val_loss', patience=8, verbose=1, min_delta=1e-4) # остановка обучения, если loss на валидационном множесте улучшается менее чем на 10^-4
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4, verbose=1, min_delta=1e-4)
    filepath=callback_folder+"/resnetm-"+timestr+"-{epoch:02d}-{val_acc:.4f}.hdf5" #
    check = ModelCheckpoint(filepath, monitor = "val_acc", save_best_only = False) # сохранение лучшей (с наибольшим acc на валидационном множестве) сети

    tb_callback = TensorBoard(log_dir=callback_folder+'/tb-logs-'+timestr, histogram_freq=0,
                              write_graph=True, write_images=False)
    callbacks_list = [early_stop, reduce_lr , check, tb_callback]
    
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
