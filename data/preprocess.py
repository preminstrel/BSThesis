import os
import numpy as np
from PIL import Image
import cv2

def resize(path):
    dir_list = os.listdir(path + 'REFUGE-Validation400/')
    for i in dir_list:
        img_path = os.path.join(path, 'REFUGE-Validation400/', i)
        img = Image.open(img_path).convert('RGB')

        newsize = (256, 256)
        img = img.resize(newsize)
        new_img_path = os.path.join(path, 'resized', i)
        img.save(new_img_path, quality=95)

def cvt2CLAHE(path, out, gridsize=8, clipLimit=2.0):
    '''
    CLAHE 1st method, with better color performance but worse channel performance
    '''
    bgr = cv2.imread(path)
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(img)
    gridsize=8
    clahe = cv2.createCLAHE(clipLimit=clipLimit,tileGridSize=(gridsize,gridsize))
    img[:,:,0] = clahe.apply(img[:,:,0])
    img = cv2.cvtColor(img, cv2.COLOR_Lab2BGR)
    cv2.imwrite(out, img)
    return img

def cvt2CLAHE2(path, out, gridsize=8, clipLimit=2.0):
    '''
    CLAHE 2nd method, with worse color performance but better channel performance
    '''
    img = cv2.imread(path)
    clahe = cv2.createCLAHE(clipLimit, tileGridSize=(gridsize,gridsize))
    img_new_1 = clahe.apply(img[:,:,0])
    img_new_2 = clahe.apply(img[:,:,1])
    img_new_3 = clahe.apply(img[:,:,2])
    img_merge = cv2.merge([img_new_1,img_new_2,img_new_3])
    cv2.imwrite(out, img_merge)
    return img

def preprocess(origin_dir, out_dir):
    for i in os.listdir(origin_dir):
        img_path = os.path.join(origin_dir, i)
        new_img_path = os.path.join(out_dir, i)
        #cvt2CLAHE(img_path, new_img_path)
        cvt2CLAHE2(img_path, new_img_path)

if __name__ == "__main__":
    # print("Start Resize images...")
    # path = '/mnt/data3_ssd/RetinalDataset/REFUGE/'
    # resize(path)
    # print(len(os.listdir(path + 'resized/')))

    print("[INIT] Start Processing images...")
    origin_dir = '/mnt/data3_ssd/RetinalDataset/ODIR-5K/train_resized/'
    out_dir = '/mnt/data3_ssd/RetinalDataset/ODIR-5K/train_CLAHE2'
    preprocess(origin_dir, out_dir)
    print(f"[DONE] imgs in {out_dir}: ", len(os.listdir(out_dir)))
    origin_dir = '/mnt/data3_ssd/RetinalDataset/ODIR-5K/valid_resized/'
    out_dir = '/mnt/data3_ssd/RetinalDataset/ODIR-5K/valid_CLAHE2'
    preprocess(origin_dir, out_dir)
    print(f"[DONE] imgs in {out_dir}: ", len(os.listdir(out_dir)))
