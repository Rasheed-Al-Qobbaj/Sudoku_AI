import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model
from puzzles import print_board

# --- Model Loading ---
try:
    model = load_model('sudoku_custom_model_with_feedback.keras')
except Exception as e:
    print(f"Error loading model: {e}")
    model = None


# --- Image Processing Functions ---
def find_biggest_contour(contours):
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
    rows = np.vsplit(warped_image, 9)
    cells = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for cell in cols:
            cells.append(cell)
    return cells


# --- Cell Preparation Function ---
def prepare_cell_for_model(cell_image):
    """
    Cleans, centers, and prepares a cell image for prediction with a much more robust pipeline.
    """
    # 1. Image Cleaning and Thresholding
    # Add a 5-pixel border to handle grid lines that bleed into the cell
    img_bordered = cv2.copyMakeBorder(cell_image, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # Use Otsu's binarization which automatically finds the optimal threshold
    # This is great for handling varying background colors (white vs gray)
    _, img_thresh = cv2.threshold(img_bordered, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # 2. Find the Contour of the Digit
    contours, _ = cv2.findContours(img_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 3. Check for Empty Cells
    if not contours:
        return None  # No contour found, this is an empty cell

    # Find the largest contour, which should be the digit
    largest_contour = max(contours, key=cv2.contourArea)

    # If the contour is too small, it's likely noise, not a digit.
    if cv2.contourArea(largest_contour) < 50:
        return None

    # 4. Center the Digit
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Create a new, square, black image to place the digit onto
    size = max(w, h)
    digit_img_centered = np.zeros((int(size * 1.5), int(size * 1.5)), dtype=np.uint8)

    # Calculate new position to paste the cropped digit
    start_x = (int(size * 1.5) - w) // 2
    start_y = (int(size * 1.5) - h) // 2

    # Paste the digit onto the center of our new black canvas
    digit_img_centered[start_y:start_y + h, start_x:start_x + w] = img_thresh[y:y + h, x:x + w]

    # 5. Final Preparation for the Model
    # Resize to 28x28, the required input size for our MNIST model
    img_resized = cv2.resize(digit_img_centered, (28, 28))

    # Add a small border and resize again to better match MNIST's format
    img_padded = cv2.copyMakeBorder(img_resized, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
    img_final = cv2.resize(img_padded, (28, 28))

    # Normalize and reshape for the model
    img_normalized = img_final / 255.0
    img_reshaped = np.reshape(img_normalized, (1, 28, 28, 1))

    return img_reshaped


def detect_and_normalize_polarity(warped_grid):
    """
    Analyzes the entire grid's histogram to robustly detect polarity and standardize it.
    Returns a grid that is always dark text on a light background.
    """
    # Calculate the median pixel value of the entire grid
    median_pixel = np.median(warped_grid)

    # If the median is dark (less than half brightness), we assume it's a "dark mode" puzzle
    if median_pixel < 127:
        # Invert the image
        return cv2.bitwise_not(warped_grid)

    # Otherwise, it's already in the standard "light mode" format
    return warped_grid


# --- Recognizer and Main Pipeline ---
def recognize_digits(cells):
    """
    Final, robust recognition pipeline. Uses a rule-based check for empty cells
    and sends cleaned, centered digits to the custom-trained model.
    """
    if not model:
        print("Model not loaded.")
        return None

    board = np.zeros((9, 9), dtype=int)
    class_names = sorted([int(d) for d in os.listdir('[SCRAPPED] CNN/dataset_augmented') if d.isdigit()])

    for i, cell in enumerate(cells):
        # 1. Use your rule-based check for empty cells first.
        img_thresh = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # Crop the border to focus on the center where a digit would be
        center_crop = img_thresh[10:40, 10:40]
        if cv2.countNonZero(center_crop) < 80:
            digit = 0
        else:
            # 2. If not empty, prepare the cell for our model
            contours, _ = cv2.findContours(img_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                digit = 0  # No contour found, treat as empty
            else:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)

                # Crop the digit tightly from the thresholded image
                digit_roi = img_thresh[y:y + h, x:x + w]

                # Create a padded, centered image to match training data
                padded_size = int(max(w, h) * 1.4)
                centered_digit = np.zeros((padded_size, padded_size), dtype=np.uint8)

                start_x = (padded_size - w) // 2
                start_y = (padded_size - h) // 2
                centered_digit[start_y:start_y + h, start_x:start_x + w] = digit_roi

                # Resize to 28x28 for the model
                final_img = cv2.resize(centered_digit, (28, 28))
                final_img = np.reshape(final_img, (1, 28, 28, 1))

                # Predict
                prediction = model.predict(final_img, verbose=0)
                predicted_class_index = np.argmax(prediction)
                digit = class_names[predicted_class_index]

        row, col = i // 9, i % 9
        board[row][col] = digit

    return board.tolist()


def image_to_board(image_path):
    """The complete pipeline with a more robust grid finding mechanism."""
    img = cv2.imread(image_path)
    if img is None: return None, None, None
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_contours = img.copy()

    # --- Robust Grid Finding ---
    # Method 1: Adaptive Threshold (Good for varied lighting)
    img_thresh_adaptive = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                                11, 2)
    contours, _ = cv2.findContours(img_thresh_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest_contour = find_biggest_contour(contours)

    # Method 2: Fallback to Simple Threshold (Good for high contrast / dark mode)
    if biggest_contour.size == 0:
        _, img_thresh_simple = cv2.threshold(img_blur, 128, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(img_thresh_simple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest_contour = find_biggest_contour(contours)

    if biggest_contour.size == 0:
        print("Could not find a Sudoku grid contour.")
        return None, img_contours, None
    # --- End of Robust Grid Finding ---

    cv2.drawContours(img_contours, [biggest_contour], -1, (0, 255, 0), 3)
    points = reorder_points(biggest_contour)
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_warped_gray = cv2.warpPerspective(img_gray, matrix, (450, 450))

    img_standardized = detect_and_normalize_polarity(img_warped_gray)

    cells = split_into_cells(img_standardized)
    board = recognize_digits(cells)  # We will replace this function next

    return board, img_contours, img_standardized


if __name__ == '__main__':

    for file in os.listdir('images'):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_file = os.path.join('images', file)
            print(f"--- Processing image: {image_file} ---")

            final_board, contour_image, warped_image = image_to_board(image_file)

            if final_board:
                print("Recognized Sudoku Board:")
                print_board(final_board)
                cv2.imshow("Original with Contour", contour_image)
                cv2.imshow("Warped Grid", warped_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("Could not process the Sudoku puzzle from this image.\n")