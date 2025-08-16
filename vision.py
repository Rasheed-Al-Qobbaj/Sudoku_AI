import cv2
import numpy as np
import os
import pytesseract

from puzzles import print_board


# --- Tesseract Configuration ---
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- Image Processing Helpers (Unchanged) ---
def find_biggest_contour(contours):
    biggest = np.array([])
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx;
                max_area = area
    return biggest


def reorder_points(points):
    points = points.reshape((4, 2));
    new_points = np.zeros((4, 1, 2), dtype=np.int32)
    add = points.sum(1);
    new_points[0] = points[np.argmin(add)];
    new_points[3] = points[np.argmax(add)]
    diff = np.diff(points, axis=1);
    new_points[1] = points[np.argmin(diff)];
    new_points[2] = points[np.argmax(diff)]
    return new_points


def split_into_cells(warped_image):
    rows = np.vsplit(warped_image, 9);
    cells = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for cell in cols: cells.append(cell)
    return cells


def detect_and_normalize_polarity(warped_grid):
    median_pixel = np.median(warped_grid)
    if median_pixel < 127: return cv2.bitwise_not(warped_grid)
    return warped_grid


# --- THE NEW RECOGNITION ENGINE ---
def recognize_digits(cells):
    board = np.zeros((9, 9), dtype=int)
    for i, cell in enumerate(cells):
        # 1. Pre-process the cell
        cell_crop = cell[10:40, 10:40]  # Crop to the center
        img_thresh = cv2.threshold(cell_crop, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # 2. Use your pixel count rule for empty cells (fast and reliable)
        if cv2.countNonZero(img_thresh) < 80:
            digit = 0
        else:
            # 3. For non-empty cells, use the power of Tesseract
            # Enlarge the original cell image for better OCR accuracy
            cell_large = cv2.resize(cell, (0, 0), fx=2, fy=2)

            # Tesseract config: --psm 10 = treat as a single character.
            # tessedit_char_whitelist = only look for these characters.
            config = r'--psm 10 -c tessedit_char_whitelist=123456789'
            text = pytesseract.image_to_string(cell_large, config=config)

            try:
                digit = int(text.strip())
            except:
                digit = 0  # If Tesseract fails, default to 0

        row, col = i // 9, i % 9
        board[row][col] = digit

    return board.tolist()


# --- THE FINAL, ROBUST MAIN PIPELINE ---
def image_to_board(image_path):
    img = cv2.imread(image_path)
    if img is None: return None, None, None
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_contours = img.copy()

    # Robust Grid Finding
    img_thresh_adaptive = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
                                                11, 2)
    contours, _ = cv2.findContours(img_thresh_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    biggest_contour = find_biggest_contour(contours)
    if biggest_contour.size == 0:
        _, img_thresh_simple = cv2.threshold(img_blur, 128, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(img_thresh_simple, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest_contour = find_biggest_contour(contours)
    if biggest_contour.size == 0: return None, img_contours, None

    cv2.drawContours(img_contours, [biggest_contour], -1, (0, 255, 0), 3)
    points = reorder_points(biggest_contour)
    pts1 = np.float32(points)
    pts2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    img_warped_gray = cv2.warpPerspective(img_gray, matrix, (450, 450))

    img_standardized = detect_and_normalize_polarity(img_warped_gray)
    cells = split_into_cells(img_standardized)
    board = recognize_digits(cells)

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