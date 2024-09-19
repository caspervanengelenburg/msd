import os
import torch
import cv2
import numpy as np
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
from tqdm import tqdm

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def add_padding(image, padding_size, color):
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image object")

    if image.mode == 'L' and isinstance(color, tuple):
        color = color[0]

    return ImageOps.expand(image, border=padding_size, fill=color)

def crop_center_np(img, cropx, cropy):
    y, x = img.shape[:2]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[starty:starty+cropy, startx:startx+cropx]

def grayscale_to_rgb(grayscale_image):
    grayscale_image = grayscale_image / 255
    rgb_image = np.stack((grayscale_image, grayscale_image, grayscale_image), axis=-1)
    return rgb_image

def process_npy_files(base_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)

    model_type = "vit_h"
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    device = "cuda:0"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam.eval()
    
    mask_generator = SamAutomaticMaskGenerator(sam)

    for file in tqdm(os.listdir(base_dir), desc='Processing files'):
        if file.endswith('.npy'):
            file_path = os.path.join(base_dir, file)
            struct_in = np.load(file_path)
            boundary_image = struct_in[..., 0].astype(np.uint8)
            rgb_image = grayscale_to_rgb(boundary_image)
            rgb_image = Image.fromarray(np.uint8(rgb_image * 255))
            color = (255, 255, 255)
            rgb_pad_image = add_padding(rgb_image, 1024, color)
            img = np.array(rgb_pad_image)

            masks = mask_generator.generate(img)

            sorted_masks = sorted(masks, key=lambda k: k['area'], reverse=True)
            cropped_image = crop_center_np(sorted_masks[0]['segmentation'], 512, 512)
            in_out_channel = np.where(cropped_image, 255, 0)

            mask = boundary_image == 0
            in_wall_out_channel = in_out_channel.copy()
            in_wall_out_channel[mask] = 128

            combined_img = np.dstack((in_wall_out_channel, in_out_channel, boundary_image))
            np.save(os.path.join(dest_dir, file), combined_img)


if __name__ == "__main__":
    folder_list = ['train', 'test']

    base_root = '../dataset/'
    dest_root = '../dataset_processed/'

    for folder in folder_list:
        base_dir = f'{base_root}{folder}/struct_in/'
        dest_dir = f'{dest_root}{folder}/struct_in/'

        # Process .npy files for the current folder
        print(f"Processing {folder} folder...")
        process_npy_files(base_dir, dest_dir)
        print(f"Finished processing {folder} folder.")
