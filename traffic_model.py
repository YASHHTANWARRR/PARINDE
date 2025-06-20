import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from ultralytics import YOLO
from PIL import Image
import shutil
import random

dataset1= [r'C:\Users\birba\OneDrive\Desktop\PARINDE\PARINDE\dataset\trafic_data\train']
dataset2= [r'C:\Users\birba\OneDrive\Desktop\PARINDE\OBJECT-DETECTION\data']
merged_folder = 'Users/birba/OneDrive/Desktop/PARINDE/PARINDE/output_merged'
datatset_paths = [[dataset1,dataset2]]

output_merged = merged_folder 
splits = ['train','labels','test']

split_ratio = [0.8,0.1,0.1]

for split in splits:
    os.makedirs(f"{output_merged}/images/{split}",exist_ok=True)
    os.makedirs(f"{output_merged}/images/{split}",exist_ok=True)
    
all_data = []
for dataset_path in datatset_paths:
    image_dir = os.path.join(dataset_path,"images")
    image_dir = os.path.join(dataset_path,"labels")
    for fname in os.listdir(image_dir):
        if fname.endswith(".jpg") :
            imag_path = os.path.join(image_dir,fname)
            label_path = os.path.join(image_dir,os.path.splitext(fname)[0]+".txt")
            if os.path.exists(label_path):
                all_data.append((imag_path,label_path))
    
random.shuffle(all_data)
num_total = len(all_data)
train_end = int(num_total * split_ratio[0])
val_end = train_end + int(num_total * split_ratio[1])

splits_data = {
    'train': all_data[:train_end],
    'val': all_data[train_end:val_end],
    'test': all_data[val_end:]
}

for split, data in splits_data.items():
    for img_path, label_path in data:
        shutil.copy(img_path, f"{output_merged}/images/{split}/")
        shutil.copy(label_path, f"{output_merged}/labels/{split}/")

print("Datasets merged and split successfully.")             
            