# Lane-detection-in-image
Lane Detection using Region of Interest (ROI) and Canny Edge Detection

**Overview**


This project implements a basic lane detection pipeline using image processing techniques such as defining a region of interest, edge detection, and Hough Line Transform. The project evaluates the performance of the lane detection by computing accuracy metrics based on a manually defined ground truth.

**Prerequisites**

Before running the code, ensure you have the following dependencies installed:

•	Python 3.x

•	OpenCV (cv2)

•	NumPy (numpy)

•	Matplotlib (matplotlib)

You can install the required libraries using:
pip install opencv-python numpy matplotlib
Steps in the Lane Detection Pipeline

**1. Importing Required Libraries**

```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
%matplotlib inline



**2. Reading the Image**

image = mpimg.imread('/content/solidWhiteCurve.jpg')
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)

**3. Mounting Google Drive (if using Colab)**

from google.colab import drive
drive.mount('/content/drive')
image = mpimg.imread('/content/drive/MyDrive/3(1).jpg')
print('This image is:', type(image), 'with dimensions:', image.shape)
plt.imshow(image)

**4. Defining the Region of Interest (ROI)**

vertices = np.array([[(0, image.shape[0]), (450, 290), (490, 290), (image.shape[1], image.shape[0])]], dtype=np.int32)
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    ignore_mask_color = (255,) * (img.shape[2] if len(img.shape) > 2 else 1)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return cv2.bitwise_and(img, mask)
masked_image = region_of_interest(image, vertices)
plt.imshow(masked_image)

**5. Edge Detection using Canny Filter**

gray_image = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
cannyed_image = cv2.Canny(gray_image, 100, 200)
plt.imshow(cannyed_image)

**6. Detecting Lines using Hough Transform**

lines = cv2.HoughLinesP(
    cannyed_image,
    rho=6,
    theta=np.pi/60,
    threshold=160,
    lines=np.array([]),
    minLineLength=40,
    maxLineGap=25
)

def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    if lines is None:
        return img
    img = np.copy(image)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    return cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)

line_image = draw_lines(cannyed_image, lines)
plt.imshow(line_image)

**7. Creating a Ground Truth Mask**

ground_truth_mask = np.zeros_like(cannyed_image)
cv2.line(ground_truth_mask, (450, 290), (490, 290), 255, 5)
cv2.line(ground_truth_mask, (0, image.shape[0]), (450, 290), 255, 5)
cv2.line(ground_truth_mask, (490, 290), (image.shape[1], image.shape[0]), 255, 5)
plt.imshow(ground_truth_mask, cmap='gray')

**8. Computing Accuracy Metrics**

def compute_accuracy(detected_mask, ground_truth_mask):
    TP = np.sum((detected_mask == 255) & (ground_truth_mask == 255))
    TN = np.sum((detected_mask == 0) & (ground_truth_mask == 0))
    FP = np.sum((detected_mask == 255) & (ground_truth_mask == 0))
    FN = np.sum((detected_mask == 0) & (ground_truth_mask == 255))
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return accuracy, precision, recall, f1_score

accuracy, precision, recall, f1_score = compute_accuracy(cannyed_image, ground_truth_mask)
print(f'Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, F1-Score: {f1_score:.2f}')

**Output**

The processed image displays detected lanes with overlaid lines and calculated accuracy metrics.

**Summary**

This project demonstrates:

•	Image preprocessing (grayscale conversion and edge detection)

•	Defining a region of interest for lane detection

•	Detecting lines using the Hough Transform

•	Overlaying detected lanes on the original image

•	Computing accuracy metrics against a manually defined ground truth


**License**

This project is open-source and can be used freely with attribution.

**Author**

R.Madhuni

