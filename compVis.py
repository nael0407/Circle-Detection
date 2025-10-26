import cv2
import numpy as np

img = cv2.imread("Photo Inspo.jpeg")
output = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)
gray = cv2.GaussianBlur(gray, (7,7), 1.5)
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
gray = clahe.apply(gray)
adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
adaptive = cv2.medianBlur(adaptive, 5)
edges = cv2.Canny(adaptive, 100, 200)


circles = cv2.HoughCircles(
    edges,
    cv2.HOUGH_GRADIENT,
    dp = 2,
    minDist = 70,
    param1 = 160,
    param2 = 61,
    minRadius = 33,
    maxRadius = 43
)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for (x, y, r) in circles[0, :]:
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        cv2.circle(output, (x, y), 2, (0, 0, 255), 3)

cv2.imshow('Edges', edges)
cv2.imshow('Gray Preprocessed', gray)
cv2.imshow('Detected Circle', output)
cv2.waitKey(0)
cv2.destroyAllWindows()