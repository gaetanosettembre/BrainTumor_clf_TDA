"""

@author: Serena Grazia De Benedictis, Grazia Gargano, Gaetano Settembre
"""

# If you use Spyder as IDE, uncomment the following lines
# import os
# os.getcwd()
# os.chdir(r'C:\Users\.....\Project_MRI') # Specify the project absolute path

import numpy as np
from utils import import_data, data_to_negative, import_data_without_preprocessing, display_image_grid
from utilsTDA import  extractROI, image_with_ROI

labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Set Training and Test set path
path_train = 'Dataset/Training'
path_test = 'Dataset/Testing'

# Import Training and Test Set
x_train, y_train = import_data(path_train, labels, 250)
x_test, y_test = import_data(path_test, labels, 250)

# Import Training and Test set without preprocessing
x_train_wp, y_train_wp = import_data_without_preprocessing(path_train, labels, 250)
x_test_wp, y_test_wp = import_data_without_preprocessing(path_test, labels, 250)

# Make images negative
x_train = data_to_negative(x_train)
x_test = data_to_negative(x_test)

# Normalization
x_train = np.array(x_train) / 255.0  # normalize Images into range 0 to 1.
x_test = np.array(x_test) / 255.0


# =============================================================================
# ROIs extraction and final TDA output from images in Dataset
# =============================================================================

ROI_train = extractROI(x_train, window_size=10, border_width=70)
ROI_test = extractROI(x_test, window_size=10, border_width=70)

Img_train_with_ROI = image_with_ROI(x_train_wp,ROI_train,param_mask=0.5)
Img_test_with_ROI = image_with_ROI(x_test_wp,ROI_test,param_mask=0.5)


# Show some images with extracted ROIs in Training dataset
images_to_display = [Img_train_with_ROI[i] for i in range(36)]
display_image_grid(images_to_display, grid_size=(6, 6), figsize=(10, 10), cmap="gray", title=None)
