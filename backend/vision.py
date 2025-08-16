import cv2
import numpy as np
import os

from puzzles import print_board
from tensorflow.keras.models import load_model

# --- Configuration ---
MODEL_PATH = '../models/final_sudoku_custom_augmented_model.keras'
DATASET_DIR = '../CNN/dataset_final_augmented'

# --- Model Loading ---
try:
    model = load_model(MODEL_PATH)
    # The model predicts an index. We need to map this index back to the actual digit.
    # We get this mapping from the folder names in the dataset directory.
    class_names = sorted([int(d) for d in os.listdir(DATASET_DIR) if d.isdigit()])
except Exception as e:
    print(f"CRITICAL ERROR: Could not load model or dataset class names.")
    print(f"Error details: {e}")
    model = None
    class_names = []


# --- Image Processing Helper Functions ---
def find_biggest_contour(contours):
    """Finds the largest contour that has 4 corner points."""
    biggest = np.array([])
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest


def reorder_points(points):
    """Reorders corner points to be [top-left, top-right, bottom-left, bottom-right]."""
    points = points.reshape((4, 2))
    new_points = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1)
    new_points[0] = points[np.argmin(add)]
    new_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]
    new_points[2] = points[np.argmax(diff)]
    return new_points


def split_into_cells(warped_image):
    """Splits the 450x450 grid image into 81 individual 50x50 cell images."""
    rows = np.vsplit(warped_image, 9)
    cells = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for cell in cols:
            cells.append(cell)
    return cells


def detect_and_normalize_polarity(warped_grid):
    """
    Robustly detects if the grid is light-on-dark or dark-on-light and standardizes it
    to always be dark text on a light background.
    """
    median_pixel = np.median(warped_grid)
    # If the median pixel value is dark, it's likely a dark-mode image. Invert it.
    if median_pixel < 127:
        return cv2.bitwise_not(warped_grid)
    return warped_grid


# --- Final Recognition Logic ---
def recognize_digits(cells):
    """
    Uses the rule-based check for empty cells and sends cleaned digits to the custom CNN model.
    """
    if not model or not class_names:
        print("Model/class names not loaded. Cannot recognize digits.")
        return None

    board = np.zeros((9, 9), dtype=int)

    for i, cell in enumerate(cells):
        # 1. Rule-based check for empty cells (fast and reliable)
        img_thresh_inv = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        center_crop = img_thresh_inv[10:40, 10:40]
        if cv2.countNonZero(center_crop) < 100:
            digit = 0
        else:
            # 2. If not empty, prepare and predict with our robust model
            contours, _ = cv2.findContours(img_thresh_inv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                digit = 0
            else:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                digit_roi = img_thresh_inv[y:y + h, x:x + w]

                # Create a padded, centered image to match training data format
                padded_size = int(max(w, h) * 1.5)
                centered_digit = np.zeros((padded_size, padded_size), dtype=np.uint8)
                start_x = (padded_size - w) // 2
                start_y = (padded_size - h) // 2
                centered_digit[start_y:start_y + h, start_x:start_x + w] = digit_roi

                # Resize to 28x28 for the model
                final_img = cv2.resize(centered_digit, (28, 28))
                final_img = np.reshape(final_img, (1, 28, 28, 1))

                # Predict the digit
                prediction = model.predict(final_img, verbose=0)
                predicted_class_index = np.argmax(prediction)
                digit = class_names[predicted_class_index]

        row, col = i // 9, i % 9
        board[row][col] = digit

    return board.tolist()


# --- Main Pipeline Function ---
def image_to_board(image_path):
    """The complete, robust pipeline from image file to a 9x9 Sudoku board list."""
    img = cv2.imread(image_path)
    if img is None: return None, None, None
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_contours = img.copy()

    # grid finding: Try adaptive threshold first, fallback to simple threshold
    img_thresh_adaptive = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                                11, 2)
    contours, _ = cv2.findContours(img_thresh_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest_contour = find_biggest_contour(contours)
    if biggest_contour.size == 0:
        _, img_thresh_simple = cv2.threshold(img_blur, 128, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(img_thresh_simple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest_contour = find_biggest_contour(contours)
        if biggest_contour.size == 0:
            return None, img_contours, None

    cv2.drawContours(img_contours, [biggest_contour], -1, (0, 255, 0), 3)
    points = reorder_points(biggest_contour)
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_warped_gray = cv2.warpPerspective(img_gray, matrix, (450, 450))

    # Standardize the image polarity (handles dark mode)
    img_standardized = detect_and_normalize_polarity(img_warped_gray)

    # Extract cells and recognize digits
    cells = split_into_cells(img_standardized)
    board = recognize_digits(cells)

    return board, img_contours, img_standardized


# --- Test Block ---
if __name__ == '__main__':

    test_image_file = os.path.join('../images', 'img_4.png')
    print(f"--- Running a test on: {test_image_file} ---")

    if os.path.exists(test_image_file):
        final_board, contour_image, standardized_grid = image_to_board(test_image_file)

        if final_board:
            print("Recognized Sudoku Board:")
            print_board(final_board)
            cv2.imshow("Original with Contour", contour_image)
            if standardized_grid is not None:
                cv2.imshow("Standardized Warped Grid", standardized_grid)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Could not process the Sudoku puzzle from this image.\n")
    else:
        print(f"Test image not found at '{test_image_file}'")