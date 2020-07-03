
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

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing import image as im
import time
from shutil import copyfile
import datetime

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing import image

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
        "--test-dir",
        default='/home/cds-y/Datasets/366_Classes_NKBVS + synthetic/test/',
        type=str,
        help="val dir"
    )
    parser.add_argument(
        "--model",
        default='/home/cds-y/Projects/cds-classification/models/resnetm-2020-07-03-13-22-17-30-0.96-selected.hdf5',
        type=str,
        help="model path"
    )
    parser.add_argument(
        "--error-logs",
        type=int,
        default=1,
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
        
def save_error_matrix(test_path, model, labels, subfolders, cls_idx, idx_cls, img_shape):
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
            x = image.load_img(image_name, target_size=img_shape)
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
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    # now = datetime.datetime.now()
    # timeset = now.strftime("%d-%m-%Y_%H:%M")
    #img_height, img_width = 96, 32
    img_height, img_width = 72, 72

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
    label_names = []
    for i in range(len(idx_cls.items())):
        label_names.append(idx_cls[i])
    
    labels = test_generator.classes

    model_name = args.model

    #loaded_model = resNetM((img_height, img_width, 3), len(labels))
    #loaded_model.load_weights(model_name)
    loaded_model = load_model(model_name)

    if args.error_logs:
        save_error_matrix(test_dir, loaded_model, labels, os.listdir(test_dir), cls_idx, idx_cls, (img_height, img_width))

    start_time = time.time()
    predictions = loaded_model.predict_generator(test_generator, verbose=1)#, steps = nb_samples)
    print("Average time to pridict one photo using generator is", (time.time() - start_time)/nb_samples)
    predictions = np.array(predictions)
    predictions = np.argmax(predictions, axis=1)
    
    from sklearn.metrics import precision_recall_fscore_support, classification_report, confusion_matrix
    
    precision, recall, fscore, support = precision_recall_fscore_support(predictions, labels, labels=np.unique(labels))
    
    print(classification_report(labels, predictions,labels=np.unique(labels), target_names=label_names))
    
if __name__=="__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
