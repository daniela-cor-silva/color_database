import cv2
import numpy as np
import os

# Processing function
def process_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    # median filter
    median_filtered = cv2.medianBlur(image, 3)

    # sharpening filter
    sharpening_kernel = np.array([[0, -0.5, 0], 
                                  [-0.5, 3, -0.5], 
                                  [0, -0.5, 0]])
    sharpened_image = cv2.filter2D(median_filtered, -1, sharpening_kernel)

    # Replace original with processed
    cv2.imwrite(image_path, sharpened_image)
    print(f"Processed and replaced: {image_path}")

folder_path = os.getcwd()

# Loop 
for file_name in os.listdir(folder_path):
    if file_name.lower().endswith(('.jpg')):
        image_path = os.path.join(folder_path, file_name)
        process_image(image_path)
