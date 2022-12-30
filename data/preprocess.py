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
    bgr = cv2.imread(path)
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(img)
    gridsize=8
    clahe = cv2.createCLAHE(clipLimit=clipLimit,tileGridSize=(gridsize,gridsize))
    img[:,:,0] = clahe.apply(img[:,:,0])
    img = cv2.cvtColor(img, cv2.COLOR_Lab2BGR)
    cv2.imwrite(out, img)

def preprocess(origin_dir, out_dir):
    for i in os.listdir(origin_dir):
        img_path = os.path.join(origin_dir, i)
        new_img_path = os.path.join(out_dir, i)
        cvt2CLAHE(img_path, new_img_path)

if __name__ == "__main__":
    # print("Start Resize images...")
    # path = '/mnt/data3_ssd/RetinalDataset/REFUGE/'
    # resize(path)
    # print(len(os.listdir(path + 'resized/')))

    print("[INIT] Start Processing images...")
    origin_dir = '/mnt/data3_ssd/RetinalDataset/Kaggle/valid_resized'
    out_dir = '/mnt/data3_ssd/RetinalDataset/Kaggle/CLAHE'
    preprocess(origin_dir, out_dir)
    print(f"[DONE] imgs in {out_dir}: ", len(os.listdir(out_dir)))