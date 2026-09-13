"""
Program 9: Mean vs Gaussian vs Median
Applies Mean, Gaussian and Median filters to the same noisy image
and saves each result separately.
"""
import cv2

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

mean_result = cv2.blur(img, (5, 5))
gaussian_result = cv2.GaussianBlur(img, (5, 5), 0)
median_result = cv2.medianBlur(img, 5)

cv2.imwrite("output_mean.png", mean_result)
cv2.imwrite("output_gaussian.png", gaussian_result)
cv2.imwrite("output_median.png", median_result)

print("Mean, Gaussian and Median filters applied and saved.")
