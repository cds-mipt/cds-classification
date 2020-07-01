
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
