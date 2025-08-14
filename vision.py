import cv2
import numpy as np
import os


def process_image(image_path):
    """
    Main function to process an image, find the Sudoku grid, and warp it.
    Returns the warped grid image and the original image with contours drawn.
    """
    # 1. Image Pre-processing
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_thresh = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # 2. Find the Grid Contour
    contours, _ = cv2.findContours(img_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = img.copy()

    biggest_contour = find_biggest_contour(contours)

    if biggest_contour.size != 0:
        # Draw the found contour on the original image for visualization
        cv2.drawContours(img_contours, [biggest_contour], -1, (0, 255, 0), 3)

        # 3. Perspective Transform (Warping)
        # Reorder points for a consistent perspective transform
        points = reorder_points(biggest_contour)

        # Define the target 450x450 square for our warped image
        pts1 = np.float32(points)
        pts2 = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])

        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        img_warped = cv2.warpPerspective(img, matrix, (450, 450))
        img_warped_gray = cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)

        return img_warped_gray, img_contours
    else:
        print("No Sudoku grid found in the image.")
        return None, img_contours


def find_biggest_contour(contours):
    """Finds the largest contour with 4 corner points, which is likely the grid."""
    biggest = np.array([])
    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 50:
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
    new_points[0] = points[np.argmin(add)]  # Top-left has the smallest sum
    new_points[3] = points[np.argmax(add)]  # Bottom-right has the largest sum

    diff = np.diff(points, axis=1)
    new_points[1] = points[np.argmin(diff)]  # Top-right has smallest difference
    new_points[2] = points[np.argmax(diff)]  # Bottom-left has largest difference

    return new_points


if __name__ == '__main__':
    for file in os.listdir('images'):
        if file.endswith('.jpg') or file.endswith('.png'):
            image_file = os.path.join('images', file)
            print(f"Processing image: {image_file}")

            warped_image, contour_image = process_image(image_file)

            if warped_image is not None:
                cv2.imshow("Original Image with Contour", contour_image)
                cv2.imshow("Warped Sudoku Grid", warped_image)

                cv2.waitKey(0)
                cv2.destroyAllWindows()