
import numpy as np
import os
import ast
import argparse
import random
import configparser
from tqdm import tqdm
import cv2
import shutil

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        default='/home/z_andrei/datasets/Traffic_ligths_data/TL_Code/config',
        type=str,
        help="config path"
    )
    return parser

def copy(source_folder, 
         dataset_folder,
         target_subfolder,
         source_subfolder,
         train_test_split):
    
    files = os.listdir(source_folder+'/'+source_subfolder)
    n = len(files)
    for i, file in enumerate(files):
        if i < int(n*train_test_split):
            shutil.copyfile(source_folder+'/'+source_subfolder+'/'+file,
                        dataset_folder+'/train/'+target_subfolder+'/'+file)
        else:
            shutil.copyfile(source_folder+'/'+source_subfolder+'/'+file,
                        dataset_folder+'/test/'+target_subfolder+'/'+file)
            
def imread_utf8(filename):
    try:
        pil_image = PIL.Image.open(filename).convert('RGB')
        open_cv_image = np.array(pil_image)
        # Convert RGB to BGR
        open_cv_image = open_cv_image[:, :, ::-1].copy()
        return open_cv_image
    except Exception as e:
        print(e)
        return None
            
def rotateImages(path):
  # for each image in the current directory
    for folder in os.listdir(path):
        for image in os.listdir(path+'/'+folder):
            img_name = path+'/'+folder+'/'+image
            img = imread_utf8(img_name)
            if img == None:
                print("Invalid name: ", img_name)
            if img.shape[0] < img.shape[1]:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                cv2.imwrite(img_name, img)
            
def create_subfolders(dataset_folder, subfolders):
    for subfolder in subfolders:
        os.mkdir(dataset_folder+'/test/'+subfolder)
        os.mkdir(dataset_folder+'/train/'+subfolder)

def main(args):
    dataset_folder = config['destinations']['target_folder']
    train_test_split = float(config['params']['train_test_split'])
    subfolders = ast.literal_eval(config['subfolders']['map_folders'])
    source_folders = ast.literal_eval(config['destinations']['source_folders'])
    rotate = int(config['params']['rotate'])
    
    if os.path.exists(dataset_folder):
        shutil.rmtree(dataset_folder)

    os.mkdir(dataset_folder)
    os.mkdir(dataset_folder+'/test/')
    os.mkdir(dataset_folder+'/train/')
    create_subfolders(dataset_folder, subfolders)
        
    for source_folder in tqdm(source_folders):
        for target_subfolder, source_subfolders in subfolders.items():
            for source_subfolder in source_subfolders:
                if os.path.exists(source_folder + '/' + source_subfolder):
                    copy(source_folder, dataset_folder, 
                         target_subfolder,
                         source_subfolder,
                        train_test_split)
    
                    
    if rotate == 1:
        rotateImages(dataset_folder+'/test/')
        rotateImages(dataset_folder+'/train/')
    
if __name__=="__main__":
    parser = build_parser()
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(args.config_path)
    main(config)
