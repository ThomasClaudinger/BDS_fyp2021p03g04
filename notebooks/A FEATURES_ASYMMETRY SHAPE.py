# TASK 1: ABC FEATURES - ASYMMETRY SHAPE
file_data = '../data/example_ground_truth.csv'
path_image = '../data/example_image'
path_mask = '../data/example_segmentation'

file_features = '../features/features.csv'

df = pd.read_csv(file_data)


image_id = list(df['image_id'])



image_list = [] 
custom_seg_list = []

image_colour = 50

# STEP 1: Read and Grayscale images
for image_colour in range(50,55):
    # reads the image
    img = cv2.imread(path_image + os.sep + image_id[image_colour] + '.jpg')
    image_colour +=1
    # calls the function for grayscaling images
    gray_image = rgb2gray(img)

    thresh = threshold_otsu(gray_image)
    binary = (gray_image < thresh)

    # appends grayscaled images to list
    image_list.append(binary)


# STEP 2: Customize masks for all gray images
# take all gray images, turn them into binary images by threshold
# append new segmentation images into list
for image_gray in range(5):
    
    # Erosion will get rid of hairs but also make the lesion smaller. 
    # Dilation will restore the lesion (but not the hairs)

    # There is some noise, we can get rid of it by morphological operators
    #Structural element, that we will use as a "brush" on our mask
    struct_el = morphology.disk(20)

    #mask_eroded = morphology.binary_erosion(mask, struct_el)

    opened = opening(image_list[image_gray],struct_el)
    img_crop = crop_image(opened)
 
    # Gaussian filtering (blur)
    blurred_mask = filters.gaussian(img_crop,sigma=5) 

    # Rotate mask
    custom_seg_list.append(blurred_mask)

# STEP 3: Save all customized segmentation mask by image_id 

path_processed = '../data/processed'
idx = 50
for img in range(5):
    #img_crop = crop_image(custom_seg_list[img])
    plt.imshow(custom_seg_list[img], cmap='gray')
    #plt.imshow(custom_seg_list[img], cmap="gray")
    plt.savefig(path_processed + os.sep + image_id[idx] + "_customized.png")
    idx +=1

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def crop_image(img,tol=0):
    # img is 2D image data
    # tol  is tolerance
    mask = img>tol
    return img[np.ix_(mask.any(1),mask.any(0))]



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