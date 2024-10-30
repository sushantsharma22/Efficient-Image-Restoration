import os
import cv2
from tqdm import tqdm
import numpy as np

# Paths to dataset directories
train_hr_dir = "/Users/sushant-sharma/Documents/Efficient-Image-Restoration/data/archive/DIV2K_train_HR"
valid_hr_dir = "/Users/sushant-sharma/Documents/Efficient-Image-Restoration/data/archive/DIV2K_valid_HR"

denoising_dir = "/Users/sushant-sharma/Documents/Efficient-Image-Restoration/data/denoising/"
super_res_dir = "/Users/sushant-sharma/Documents/Efficient-Image-Restoration/data/super_resolution/"
original_dir = "/Users/sushant-sharma/Documents/Efficient-Image-Restoration/data/original/"

# Create directories for output
os.makedirs(denoising_dir, exist_ok=True)
os.makedirs(super_res_dir, exist_ok=True)
os.makedirs(original_dir, exist_ok=True)

def add_noise(image):
    """Add Gaussian noise to an image."""
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

def downsample_image(image, scale=0.5):
    """Downsample an image for super-resolution tasks."""
    h, w = image.shape[:2]
    new_size = (int(w * scale), int(h * scale))
    low_res_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    return cv2.resize(low_res_image, (w, h), interpolation=cv2.INTER_LINEAR)

def prepare_dataset(input_dir, output_dir, task='denoising'):
    """Prepare dataset for a specific task."""
    for img_name in tqdm(os.listdir(input_dir)):
        img_path = os.path.join(input_dir, img_name)
        image = cv2.imread(img_path)

        if task == 'denoising':
            processed_image = add_noise(image)
        elif task == 'super_resolution':
            processed_image = downsample_image(image)
        else:
            continue

        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, processed_image)

# Prepare datasets
print("Preparing noisy dataset for denoising...")
prepare_dataset(train_hr_dir, denoising_dir, task='denoising')

print("Preparing low-resolution dataset for super-resolution...")
prepare_dataset(train_hr_dir, super_res_dir, task='super_resolution')
