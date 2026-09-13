"""
Program 8: Salt-and-Pepper Noise Reduction
Reads an image containing salt-and-pepper (impulse) noise and applies
median filtering to remove it.
"""
import cv2

# Input image already contains visible salt-and-pepper noise
img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

# Median filter is well suited for impulse (salt-and-pepper) noise
# because it replaces each pixel with the median of its neighborhood,
# which discards extreme outlier (0 or 255) values.
filtered = cv2.medianBlur(img, 5)

cv2.imwrite("output.png", filtered)
print("Median filtering applied to remove salt-and-pepper noise.")
