import cv2
import numpy as np
import os
import argparse

# args
parser = argparse.ArgumentParser(description="Generate binary masks")
parser.add_argument("--labeldir", type=str, required=True, help="YOLO labels")
parser.add_argument("--imdir", type=str, required=True, help="Input images")
parser.add_argument("--maskdir", type=str, required=True, help="Output")
args = parser.parse_args()

labeldir = args.labeldir
imdir = args.imdir
maskdir = args.maskdir

for imfile in os.listdir(imdir):
    if imfile.endswith((".jpg", ".png")):  
        image_path = os.path.join(imdir, imfile)
        label_path = os.path.join(labeldir, os.path.splitext(imfile)[0] + ".txt")
        
        if not os.path.exists(label_path):
            print(f"Label file not found for {imfile}")
            continue

        im = cv2.imread(image_path)
        if im is None:
            print(f"Error loading image: {imfile}")
            continue

        # grayscale and binary threshold (to isolate the bird)
        img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        _, imb = cv2.threshold(img_gray, 240, 255, cv2.THRESH_BINARY_INV)

        # draw contours
        contours, _ = cv2.findContours(imb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        mask = np.zeros_like(imb)
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

        # exclude yolo boxes
        with open(label_path, 'r') as file:
            for line in file:
                if line.strip():
                    try:
                        item, xmin, ymin, xmax, ymax, conf = line.split(' ')
                        xmin, ymin, xmax, ymax = map(int, [round(float(x)) for x in [xmin, ymin, xmax, ymax]])
                        mask[ymin:ymax, xmin:xmax] = 0  # clear
                    except ValueError as e:
                        print(f"Error processing line '{line}': {e}")

        # keep only largest object
        new_mask = np.zeros_like(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(new_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # Exclude bounding boxes from the new mask
        with open(label_path, 'r') as file:
            for line in file:
                if line.strip():
                    try:
                        item, xmin, ymin, xmax, ymax, conf = line.split(' ')
                        xmin, ymin, xmax, ymax = map(int, [round(float(x)) for x in [xmin, ymin, xmax, ymax]])
                        new_mask[ymin:ymax, xmin:xmax] = 0  # Clear areas covered by bounding boxes
                    except ValueError as e:
                        print(f"Error processing line '{line}': {e}")

        # Save the final mask image
        mask_output_path = os.path.join(maskdir, f"mask_{os.path.splitext(imfile)[0]}.png")
        cv2.imwrite(mask_output_path, new_mask)
        print(f"Saved mask for {imfile} at {mask_output_path}")

print("Processing complete.")
