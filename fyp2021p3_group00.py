"""
Main script for the FYP 2021 project 3
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score


import fyp2021p3_group00_functions as util

file_data = 'data/example_ground_truth.csv'
path_image = 'data/example_image'
path_mask = 'data/example_segmentation'

file_features = 'features/features.csv'



df = pd.read_csv(file_data)


image_id = list(df['image_id'])
is_melanoma = np.array(df['melanoma'])
is_keratosis = np.array(df['seborrheic_keratosis'])


num_images = len(image_id)

features_area = np.empty([num_images,1])
features_area[:] = np.nan
features_perimeter = np.empty([num_images,1])
features_perimeter[:] = np.nan

for i in np.arange(num_images):
    
    # Define filenames related to this image
    file_image = path_image + os.sep + image_id[i] + '.jpg'
    file_mask = path_mask + os.sep + image_id[i] + '_segmentation.png'
    
    # Read the images with these filenames
    im = plt.imread(file_image)
    mask = plt.imread(file_mask)
    
    # Measure features
    a, p = util.measure_area_perimeter(mask)
    
    # Store in the variables we created before
    features_area[i,0] = a
    features_perimeter[i,0] = p
    
    ###### TODO - Here you should measure and store some other features



# Store these features so you can reuse them later
feature_data = {"id": image_id, 
                "area": features_area.flatten(),
                "perimeter": features_perimeter.flatten()
                }

df_features = pd.DataFrame(feature_data)
df_features.to_csv(file_features, index=False)    
 
    
 
# Load the data you saved, then do some analysis
df_features = pd.read_csv(file_features)
image_id = list(df_features['id'])
features_area = np.array(df_features['area'])
features_perimeter = np.array(df_features['perimeter'])

# Display the features measured in a scatterplot
axs = util.scatter_data(features_area, features_perimeter, is_melanoma)
axs.set_xlabel('X1 = Area')
axs.set_ylabel('X2 = Perimeter')
axs.legend()



# Load features and labels
x = df_features.iloc[:,1:].to_numpy()
y = is_melanoma


#Prepare cross-validation
n_splits=5
kf = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.4, random_state=1)

acc_val = np.empty([n_splits,1])
acc_test = np.empty([n_splits,1])

index_fold = 0

#Parameter for nearest neighbor classifier
k = 5

# Predict labels for each fold using the KNN algortihm
for train_index, test_val_index in kf.split(x, y):
    
    
    # split dataset into a train, validation and test dataset
    test_index, val_index = np.split(test_val_index, 2)
    
    x_train, x_val, x_test = x[train_index], x[val_index], x[test_index]
    y_train, y_val, y_test = y[train_index], y[val_index], y[test_index]
    
      
    y_pred_val, y_pred_test = util.knn_classifier(x_train, y_train, x_val, x_test, k)
    
    acc_val[index_fold] = accuracy_score(y_val,y_pred_val)
    acc_test[index_fold] = accuracy_score(y_test,y_pred_test)
   
    index_fold += 1
    
print(acc_val)
print(acc_test)
