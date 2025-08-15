import os
import shutil
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.transform import warp, AffineTransform
from skimage.util import random_noise

# --- Configuration ---
IMG_SIZE = 28
NUM_SAMPLES_PER_DIGIT = 2500
OUTPUT_DIR = 'dataset_final_augmented_with_feedback'


FONTS = [
    'arial.ttf',
    'cour.ttf',
    'times.ttf',
    'calibri.ttf',
    'verdana.ttf',
    'georgia.ttf',
    'tahoma.ttf',
    'Roboto-Regular.ttf'
]


def generate_hyper_augmented_image(digit, font_path):
    canvas_size = int(IMG_SIZE * 2)
    image = Image.new('L', (canvas_size, canvas_size), color=0)
    draw = ImageDraw.Draw(image)

    font_size = np.random.randint(28, 35)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    offset_x, offset_y = np.random.randint(-4, 4, size=2)
    draw.text((canvas_size / 4 + offset_x, canvas_size / 4 + offset_y), str(digit), fill=255, font=font)

    img_array = np.array(image, dtype=np.float32)

    # Augmentations
    shear_val = np.random.uniform(-0.3, 0.3)
    tform = AffineTransform(shear=shear_val)
    img_array = warp(img_array, tform, mode='edge')

    angle = np.random.uniform(-20, 20)
    img_array = warp(img_array, AffineTransform(rotation=np.deg2rad(angle)), mode='edge')

    img_array = (img_array * 255).astype(np.uint8)

    kernel_size = np.random.choice([1, 2, 3])
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    op = np.random.choice([cv2.dilate, cv2.erode])
    img_array = op(img_array, kernel, iterations=1)

    img_array = (random_noise(img_array, mode='gaussian', var=np.random.uniform(0.01, 0.03)) * 255).astype(np.uint8)

    contours, _ = cv2.findContours(img_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    img_cropped = img_array[y:y + h, x:x + w]
    final_image = cv2.resize(img_cropped, (IMG_SIZE, IMG_SIZE))

    return Image.fromarray(final_image)


if __name__ == '__main__':
    if os.path.exists(OUTPUT_DIR):
        print(f"Removing old dataset directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    for digit in range(1, 10):
        digit_dir = os.path.join(OUTPUT_DIR, str(digit))
        os.makedirs(digit_dir)

        print(f"Generating hyper-augmented images for digit: {digit}...")
        for i in range(NUM_SAMPLES_PER_DIGIT):
            font_path = np.random.choice(FONTS)
            img = generate_hyper_augmented_image(digit, font_path)
            if img:
                img.save(os.path.join(digit_dir, f'digit_{i}.png'))

    print("\nFinal augmented dataset generated successfully!")