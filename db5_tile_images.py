import argparse
import cv2
import numpy as np
import os

# args
parser = argparse.ArgumentParser(description="Simplify images in 10x10 tiles.")
parser.add_argument("--body_masks", type=str, required=True, help="Binary masks")
parser.add_argument("--images", type=str, required=True, help="Input images")
parser.add_argument("--tiled_images_jpg", type=str, required=True, help="Tiled images (for visual confirmation only)")
parser.add_argument("--tiled_images_txt", type=str, required=True, help="Text files")
args = parser.parse_args()

# Define directories
body_masks = args.body_masks
images = args.images
tiled_images_jpg = args.tiled_images_jpg
tiled_images_txt = args.tiled_images_txt

# Create output directories if they don't exist
os.makedirs(tiled_images_jpg, exist_ok=True)
os.makedirs(tiled_images_txt, exist_ok=True)

tile_size = (10, 10)

# Split image into tiles
def image_to_tiles(im, tile_size):
    h, w, c = im.shape
    tiles = []
    for y in range(0, h, tile_size[1]):
        for x in range(0, w, tile_size[0]):
            tile = im[y:y+tile_size[1], x:x+tile_size[0]]
            tiles.append((x, y, tile))
    return tiles

# Calculate median color in RGB, ignoring black pixels
def median_rgb_color(tile, mask_tile):
    valid_pixels = tile[mask_tile > 0]  # Only consider pixels where mask is non-zero
    if len(valid_pixels) == 0:
        return np.array([0, 0, 0])  # Default to black if no valid pixels
    return np.median(valid_pixels, axis=0).astype(int)

# Processing loop
for img in os.listdir(images):
    img_name, img_ext = os.path.splitext(img)
    
    # Define paths
    rgb_img_path = os.path.join(tiled_images_jpg, f'tiled_{img_name}.png')
    rgb_txt_path = os.path.join(tiled_images_txt, f'median_colors_rgb_{img_name}.txt')

    img_path = os.path.join(images, img)
    mask_path = os.path.join(body_masks, f'mask_{img_name}.png')

    # Load image and mask
    im = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if im is None:
        print(f"Warning: Could not load image {img_path}. Skipping.")
        continue
    if mask is None:
        print(f"Warning: Could not load mask {mask_path}. Skipping.")
        continue

    print(f"Processing image: {img_path}")

    # Apply mask to the image
    masked_image = cv2.bitwise_and(im, im, mask=mask)

    # Prepare blank tiled image
    imtrgb = np.zeros_like(im)

    # Split into tiles and calculate median colors
    rgb_tiles = image_to_tiles(masked_image, tile_size=tile_size)
    median_colors_rgb = []

    for x, y, tile in rgb_tiles:
        mask_tile = mask[y:y+tile_size[1], x:x+tile_size[0]]
        med_rgb = median_rgb_color(tile, mask_tile)

        if not np.array_equal(med_rgb, [0, 0, 0]):  # Ignore black tiles
            median_colors_rgb.append(med_rgb[::-1])  # Convert BGR to RGB
            imtrgb[y:y+tile_size[1], x:x+tile_size[0]] = med_rgb  # Fix coordinate order

    # Save the tiled image
    cv2.imwrite(rgb_img_path, imtrgb)
    
    # Save median colors
    with open(rgb_txt_path, 'w') as r:
        for median_color in median_colors_rgb:
            r.write(f'{median_color[0]} {median_color[1]} {median_color[2]}\n')

    print(f"Saved RGB median colors to {rgb_txt_path}")

print("Processing complete for all images.")
