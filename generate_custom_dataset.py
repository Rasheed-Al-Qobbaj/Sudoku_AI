import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.transform import warp, AffineTransform, SimilarityTransform
from skimage.util import random_noise

# --- Configuration ---
IMG_SIZE = 28
NUM_SAMPLES_PER_DIGIT = 2000
OUTPUT_DIR = 'dataset_final_augmented'

FONTS = ['arial.ttf',
         'cour.ttf',
         'times.ttf',
         'calibri.ttf',
         'verdana.ttf',
         'Roboto-Regular.ttf'
        ]


def generate_hyper_augmented_image(digit, font_path):
    """Generates a digit with highly realistic, aggressive augmentations."""
    canvas_size = int(IMG_SIZE * 2)
    image = Image.new('L', (canvas_size, canvas_size), color=0)
    draw = ImageDraw.Draw(image)

    # 1. Draw the digit with random size and slight position offset
    font_size = np.random.randint(25, 32)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    offset_x = np.random.randint(-3, 3)
    offset_y = np.random.randint(-3, 3)
    draw.text((canvas_size / 4 + offset_x, canvas_size / 4 + offset_y), str(digit), fill=255, font=font)

    img_array = np.array(image, dtype=np.float32)

    # 2. Augmentations
    # Perspective Warp / Shear
    shear_val = np.random.uniform(-0.15, 0.15)
    tform = AffineTransform(shear=shear_val)
    img_array = warp(img_array, tform, mode='edge')

    # Rotation
    angle = np.random.uniform(-15, 15)
    img_array = warp(img_array, SimilarityTransform(rotation=np.deg2rad(angle)), mode='edge')

    # Convert back to uint8 for OpenCV
    img_array = (img_array * 255).astype(np.uint8)

    # Varying line thickness (Morphological operations)
    kernel_size = np.random.choice([1, 2])
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    if np.random.rand() > 0.5:
        img_array = cv2.dilate(img_array, kernel, iterations=1)
    else:
        img_array = cv2.erode(img_array, kernel, iterations=1)

    # Add Noise
    img_array = (random_noise(img_array, mode='gaussian', var=0.01) * 255).astype(np.uint8)

    # 3. Final processing
    # Find the bounding box of the noisy digit
    contours, _ = cv2.findContours(img_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    # Crop and resize
    img_cropped = img_array[y:y + h, x:x + w]
    final_image = cv2.resize(img_cropped, (IMG_SIZE, IMG_SIZE))

    return Image.fromarray(final_image)


if __name__ == '__main__':
    if os.path.exists(OUTPUT_DIR):
        import shutil

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