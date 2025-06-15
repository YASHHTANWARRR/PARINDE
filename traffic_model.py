import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from ultralytics import YOLO
from PIL import Image

dataset1= 'Users/birba/OneDrive/Desktop/PARINDE/PARINDE/dataset/trafic_data'
dataset2= 'Users/birba/OneDrive/Desktop/PARINDE/OBJECT-DETECTION/data'
merged_folder = 'Users/birba/OneDrive/Desktop/PARINDE/PARINDE/output_merged'
datatset_paths = [dataset1,dataset2]

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
            