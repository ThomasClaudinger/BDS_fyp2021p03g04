# TASK 1: ABC FEATURES - ASYMMETRY SHAPE

def centerpoint(mask):
    borders = np.where(mask != 0) # This will return 2 arrays with the index where the pixels are ones
    up, down, left, right = max(borders[0]), min(borders[0]), min(borders[1]), max(borders[1])
    center = ((up+down) //2, (left + right) //2) # Tuple with the coordinates for the center of the lesion
    return center


## Horizontal Asymmetry Analysis for Customized Masks


def horizontal_asymmetry_left_plot(left_horizontal_image, right_flipped_image, left_overlapping_images):
    fig, axes = plt.subplots(1, 3, figsize=(6, 8))

    ax = axes.flatten()
    ax[0].imshow(left_horizontal_image, cmap="gray")
    ax[0].set_axis_off()
    ax[0].set_title("A: Left Horizontal Part of Image", fontsize=12, c = 'w')

    ax[1].imshow(right_flipped_image, cmap="gray")
    ax[1].set_axis_off()
    ax[1].set_title("B: Flipped Horizontal Part of Image", fontsize=12, c = 'w')

    ax[2].imshow(left_overlapping_images, cmap="gray")
    ax[2].set_axis_off()
    ax[2].set_title("Intersection of A and B:\nGray area are non-overlaps", fontsize=12, c = 'w')

    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=2.0, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.4)
    

########################

def horizontal_asymmetry_right_plot(right_image, left_flipped_image, right_overlapping_images):
    fig, axes = plt.subplots(1, 3, figsize=(6, 8))
    ax = axes.flatten()

    ax[0].imshow(right_image, cmap="gray")
    ax[0].set_axis_off()
    ax[0].set_title("C: Right Horizontal Part of Image", fontsize=12, c = 'w')

    ax[1].imshow(left_flipped_image, cmap="gray")
    ax[1].set_axis_off()
    ax[1].set_title("D: Left Flipped Part of Image", fontsize=12, c = 'w')

    ax[2].imshow(right_overlapping_images, cmap="gray")
    ax[2].set_axis_off()
    ax[2].set_title("Intersection of C and D:\nGray area are non-overlaps", fontsize=12, c = 'w')

    plt.subplots_adjust(left=0.1,
                        bottom=0.1, 
                        right=2.0, 
                        top=0.9, 
                        wspace=0.4, 
                        hspace=0.4)
    #plt.show()

def right_horizontal_symmetry(right_horizontal_image, left_flipped_image):
    img_bwa = cv2.bitwise_and(right_horizontal_image,left_flipped_image)
    img_bwo = cv2.bitwise_or(right_horizontal_image,left_flipped_image)
    img_bwx = cv2.bitwise_xor(right_horizontal_image,left_flipped_image)
    symmetry_right = np.count_nonzero(img_bwa)
    total_size_right = np.count_nonzero(img_bwo)
    asymmetry_right = np.count_nonzero(img_bwx)
    return symmetry_right, total_size_right, asymmetry_right

def left_horizontal_symmetry(left_horizontal_image, right_flipped_image):
    img_bwa = cv2.bitwise_and(left_horizontal_image,right_flipped_image)
    img_bwo = cv2.bitwise_or(left_horizontal_image,right_flipped_image)
    img_bwx = cv2.bitwise_xor(left_horizontal_image,right_flipped_image)
    symmetry_left = np.count_nonzero(img_bwa)
    total_size_left = np.count_nonzero(img_bwo)
    asymmetry_left = np.count_nonzero(img_bwx)
    return symmetry_left, total_size_left, asymmetry_left

## Calls the function to find center points
center_cust_mask = centerpoint(cust_mask_orignal)

# Displays center point coordinates
print('Center point coordinates of masked image is:', center_cust_mask)

# Left part of center line
left_horizontal_cust = cust_mask_orignal[:,0:center_cust_mask[0]+1].astype(float)

# right part mirrored over the center line
right_flipped_cust = np.fliplr(cust_mask_orignal)[:,0:center_cust_mask[0]+1].astype(float)

overlap_left_cust = cv2.addWeighted(left_horizontal_cust, 0.5, right_flipped_cust, 0.5, 1.0)


# Right part of the center line
right_horizontal_cust = cust_mask_orignal[:,center_cust_mask[0]:].astype(float)

# Left part mirrored over the center line
left_flipped_cust = np.fliplr(cust_mask_orignal)[:,center_cust_mask[0]:].astype(float)

overlapping_right_horizontal_cust = cv2.addWeighted(right_horizontal_cust, 0.5, left_flipped_cust, 0.5, 1.0)

# Calling the function: overlapping images
horizontal_asymmetry_right_plot(right_horizontal_cust, left_flipped_cust, overlapping_right_horizontal_cust)

horizontal_asymmetry_left_plot(left_horizontal_cust, right_flipped_cust, overlap_left_cust)

# Calling the function: calculate intersection, union and symmertry difference 
intersection_right_horizontal_cust, union_right_cust, symmetric_diff_right_horizontal_cust = right_horizontal_symmetry(right_horizontal_cust, left_flipped_cust)

intersection_left_horizontal_cust, union_left_cust, symmetric_diff_left_horizontal_cust = left_horizontal_symmetry(left_horizontal_cust, right_flipped_cust)