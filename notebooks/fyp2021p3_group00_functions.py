"""
Functions for the FYP 2021 project 3
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage import morphology
from scipy.spatial.distance import cdist
from scipy.stats.stats import mode


def scatter_data(x1, x2, y, ax=None):
    # scatter_data displays a scatterplot of featuress x1 and x2, and gives each point
    # a different color based on its label in y

    class_labels, indices1, indices2 = np.unique(y, return_index=True, return_inverse=True)
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111)
        ax.grid()

    colors = cm.rainbow(np.linspace(0, 1, len(class_labels)))
    for i, c in zip(np.arange(len(class_labels)), colors):
        idx2 = indices2 == class_labels[i]
        lbl = 'Class ' + str(i)
        ax.scatter(x1[idx2], x2[idx2], color=c, label=lbl)

    return ax


def measure_area_perimeter(mask):
    # Measure area: the sum of all white pixels in the mask image
    area = np.sum(mask)

    # Measure perimeter: first find which pixels belong to the perimeter.
    struct_el = morphology.disk(1)
    mask_eroded = morphology.binary_erosion(mask, struct_el)
    image_perimeter = mask - mask_eroded

    # Now we have the perimeter image, the sum of all white pixels in it
    perimeter = np.sum(image_perimeter)

    return area, perimeter


def knn_classifier(x_train, y_train, x_validation, x_test, k):
    # Returns the labels for test_data, predicted by the k-NN clasifier trained on X_train and y_train
    # Input:
    # X_train - num_train x num_features matrix with features for the training data
    # y_train - num_train x 1 vector with labels for the training data
    # X_validation - num_test x num_features matrix with features for the validation data
    # X_test - num_test x num_features matrix with features for the test data
    # k - Number of neighbors to take into account
    # Output:
    # y_pred_validation - num_test x 1 predicted vector with labels for the validation data
    # y_pred_test - num_test x 1 predicted vector with labels for the test data

    x_test_val = np.vstack((x_validation, x_test))
    
    # Compute standardized euclidian distance of validation and test points to the other points
    distances = cdist(x_test_val, x_train, metric='seuclidean')
    
    # Sort distances per row and return array of indices from low to high
    sort_ix = np.argsort(distances, axis=1)
    
    # Get the k smallest distances
    sort_ix_k = sort_ix[:, :k]
    predicted_labels = y_train[sort_ix_k]
    
    # Predictions for each point is the mode of the K labels closest to the point
    predicted_labels = mode(predicted_labels, axis=1)[0]
    y_pred_validation = predicted_labels[:len(x_validation)]
    y_pred_test = predicted_labels[len(x_validation):]
    
    return y_pred_validation, y_pred_test
