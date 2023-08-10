# -*- coding: utf-8 -*-
"""
Created on Fri May 12 19:49:15 2023

@authors:
    Taofik Ahmed Suleiman
    Daniel Tweneboah Anyimadu
    Andrew Dwi Permana
    Hsham Abdalgny Abdalwhab Ngim

@supervisor: 
    Professor Alessandra Scotto di Freca
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from glob import glob
import seaborn as sns
from PIL import Image
import random
import cv2
np.random.seed(42)
from sklearn.utils import resample


# Load the groundtruth, and add the benign column.   
skin_data = pd.read_csv('Skin/ISIC-2017_Training_Part3_GroundTruth.csv')
skin_data['benign'] = ((skin_data['melanoma'] == 0) & (skin_data['seborrheic_keratosis'] == 0)).astype(int)

# To perform 2-step hierarchical binary classification:
# we create a new column "others" and initialize it to zero, and then set the value of "others" to one for instances with melanoma or seborrheic_keratosis
skin_data['others'] = 0
skin_data.loc[(skin_data['melanoma'] == 1) | (skin_data['seborrheic_keratosis'] == 1), 'others'] = 1

# Because of the high class imbalance, downsample the benign and upsample others class to 1000 samples
benign_downsampled = resample(skin_data[skin_data['benign'] == 1], 
                              replace=False, n_samples=1000, random_state=42)
others_upsampled = resample(skin_data[skin_data['others'] == 1], 
                              replace=True, n_samples=1000, random_state=42)

benign_others_data = pd.concat([benign_downsampled, others_upsampled])

# Now, we shuffle the dataset to avoid any learning order bias
benign_others_data = benign_others_data.sample(frac=1, random_state=42)
print("Total number of instances:", len(benign_others_data))


''' Loading and preprocessing the image dataset

Steps:
1. Reduce the size of the images to 64 x 64 to reduce the computational complexity and make sure all samples are thesame size
2. Load the images and create a list of image paths using glob
3. Create a dictionary to map image IDs to image paths
4. Define the path and add it as a new column and then use the path to read images as pixels value
5. Normalize the image pixels for each sample '''

SIZE = 64
image_dir = "Skin/ISIC-2017_Training_Data/"
file_pattern = os.path.join(image_dir, '*.jpg')
image_paths = glob(file_pattern)
image_id_to_path = {os.path.splitext(os.path.basename(path))[0]: path for path in image_paths}

benign_others_data['image_path'] = benign_others_data['image_id'].map(image_id_to_path)
benign_others_data['image_pixels'] = benign_others_data['image_path'].map(lambda x: np.asarray(Image.open(x).resize((SIZE, SIZE))))

image_pixel = np.asarray(benign_others_data['image_pixels'].tolist())
image_pixel = image_pixel/255.
