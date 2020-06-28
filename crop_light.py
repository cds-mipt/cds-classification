import os
import pandas as pd
import argparse
import numpy as np
import cv2
from tqdm import tqdm


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-folder",
        default='/datasets/DTLD/',
        type=str,
        help="folder with images"
    )
    parser.add_argument(
        "--output-folder-test",
        default='/datasets/DTLD/DTLD_crop/test',
        type=str,
    )
    parser.add_argument(
        "--output-folder-train",
        default='/datasets/DTLD/DTLD_crop/train',
        type=str,
    )
    parser.add_argument(
        "--attitude",
        default='0.8',
        type=str,
    )
    parser.add_argument(
        "--input-file",
        default='/datasets/DTLD/JSONS/DTLD_all.json',
        type=str,
    )
    return parser

pos = [0] * 6
numb = [0] * 6
col = [0] * 6


def create_dir(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)


def create_dirs(folders):
    global folder_name
    inner_folders = [5, 5, 5]
    for i in range(len(folders)):
        folder = folder_name + '/' + folders[i]
        if not os.path.exists(folder):
            os.mkdir(folder)
            for j in range(0, inner_folders[i] + 1):
                os.mkdir(folder + '/' + str(j))
        else:
            for j in range(0, inner_folders[i] + 1):
                if not os.path.exists(folder + '/' + str(j)):
                    os.mkdir(folder + '/' + str(j))


num = 0

def check_it(i, id):
    idx = [0, 4, 3]
    global pos, numb, col, process
    if process == 'train':
        attitude = float(args.attitude)
    else:
        attitude = 1 - float(args.attitude)
 #   print(pos, numb, col)
    if i == 0:
        if pos[int(id[idx[i]])] > int(25000 * attitude):
            return 0
        else:
            pos[int(id[idx[i]])] += 1
            return 1
    elif i == 1:
        if numb[int(id[idx[i]])] > int(20000 * attitude):
            return 0
        else:
            numb[int(id[idx[i]])] += 1
            return 1
    else:
        if col[int(id[idx[i]])] > int(20000 * attitude):
            return 0
        else:
            col[int(id[idx[i]])] += 1
            return 1
def save(image, id, unique_id, filename):
    global num, folder_name, pos, numb, col
    id = str(id)
    folders = ['position', 'colours', 'lights']
    idx = [0, 4, 3]
    create_dirs(folders)
    for i, elem in enumerate(folders):
        if check_it(i, id) == 0:
            continue

        cv2.imwrite(folder_name + '/' + \
                    elem + '/' + \
                    id[idx[i]] + '/' + \
                    filename + '__' + \
                    str(unique_id)+ '.jpg', image)
        '''
        print(args.output_folder + '/' + \
                    elem + '/' + \
                    id[idx[i]] + '/' + \
                    filename + '__' + \
                    str(unique_id)+ '.jpg')
        '''
        num += 1


folder_name = ''


def crop(image, object, filename):
    for i in range(len(object)):
        if object[i]['unique_id'] is -1:
            continue
        light = image.copy()
        x, y, w, h = object[i]['x'], object[i]['y'], object[i]['width'], object[i]['height']
        if w < 20 or h < 20 or y < 0 or x < 0 or w > 200 or h > 200:
            continue
        im = np.zeros(shape=(int(h+0.2*h), int(w+0.2*w), 3))
        h, w = light[y:y+h, x:x+w,:].shape[:-1]
        #print(int(0.1*h), int(w+0.1*w), x, x+w, w)

        im = light[y-int(0.1*h):y+int(1.1*h), x-int(0.1*w):x+int(1.1*w),:]

        save(im, object[i]['class_id'], object[i]['unique_id'], filename)

        w = np.random.randint(10, 50)
        att = np.random.randn()/10 + 3
        h = int(w*att)

        x, y = np.random.randint(200, 1800), np.random.randint(400, 700)
        #print(x, y, h, w, filename)
        im = light[y:y+h, x:x+w,:]
        if im.size != 0:
            save(im, 555555, object[i]['unique_id'], filename)


process = 'train'

def del_please(folder):
    os.system("rm -rf "+folder+"/lights/0")
    os.system("rm -rf "+folder+"/position/0")
    for f in os.listdir(folder):
        for ff in os.listdir(folder+"/"+f):
            for imm in os.listdir(folder+"/"+f+"/"+ff+"/"):
                if os.path.getsize(folder+"/"+f+"/"+ff+"/"+imm) < 1:
                    os.remove(folder+"/"+f+"/"+ff+"/"+imm)


def main(args):
    global folder_name, process, pos, numb, col
    create_dir(args.output_folder_train)
    create_dir(args.output_folder_test)
    folder_name = args.output_folder_train
    df = pd.read_json(args.input_file)
    limit = float(args.attitude) * float(df.shape[0])
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        if i >= int(limit):
            folder_name = args.output_folder_test
            process = 'test'
            pos = [0] * 6
            numb = [0] * 6
            col = [0] * 6
        object = row['objects']
        path = row['path'].replace("/scratch/fs2/", args.input_folder).replace("tiff", "png")
        filename = path
#        print(row['path'])
        image = cv2.imread(filename)
        filename = filename[26:].replace("/", "_").split('.')[0]
        crop(image, object, filename)
    del_please(args.output_folder_train)
    del_please(args.output_folder_test)

if __name__=="__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
