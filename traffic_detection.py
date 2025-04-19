import numpy as np 
import pandas as pd 
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt 
from matplotlib import Rectangle 
import os 
import shutil
from PTL import Image 
import time 
import yaml 
from picamera2 import Picamera2
from tqdm import tqdm 
from sklearn.model_selection import train_test_split 

#data for training (change the directroy loactions as per  uyour device)
valid_dir = ' '
output_dir = ' '
epochs = 50
batch_size = 10 
imgsz = 640 
    
yaml_dict = {
    'train' : os.path.abspath(os.path.join(output_dir,'images','train')),
    'val': os.path.abspath(os.path.join(output_dir,'images','val'))
    'nc' : len(class_names),
    'names' : class_names
}

yaml_path = os.path.join(output_dir,'data.yaml')
with open (yaml_path ,'w') as f :
    yaml.dump(yaml_dict, f)
    
model = YOLO('yolov8n.pt')
model.train(data = yaml_path,epochs = epochs ,batch= batch_size,imgsz=imgsz)
print("Training complete")

#real time detction
best_model_path = ' '
print("loading trainig model from: {best_model_path}")
trained_model = YOLO(best_model_path)

picam2= Picamera2()
picam2.configure( picam2.create_video_configuration(main={"size" : (640,420)}))
picam2.start()
time.sleep(2)

try: 
    while True:
        start_time = time.time()
        frame = picam2.capture_array()
        results = trained_model.predict(source=frame,imgsz=imgsz,conf=0.25)
        im_result = results[0].plot()
        cv2.imshow("YOLOv8 Detection",im_result)
        print(f"FPS: {1.0 / (time.time()-start_time):.2f}")
        if cv2.waitKey(1) & 0*FF == ord('q'):
            break 

finally :
    picam2.stop()
    cv2.destroyAllWindows()
    print("detction stopped")