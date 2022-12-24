import os
import numpy as np
from PIL import Image

def resize(path):
    dir_list = os.listdir(path + 'REFUGE-Validation400/')
    for i in dir_list:
        img_path = os.path.join(path, 'REFUGE-Validation400/', i)
        img = Image.open(img_path).convert('RGB')

        newsize = (256, 256)
        img = img.resize(newsize)
        new_img_path = os.path.join(path, 'resized', i)
        img.save(new_img_path, quality=95)

if __name__ == "__main__":
    print("Start Resize images...")
    path = '/mnt/data3_ssd/RetinalDataset/REFUGE/'
    resize(path)
    print(len(os.listdir(path + 'resized/')))
