"""
Program 2: Controlled Brightness Enhancement
Reads a dark image and increases brightness by a fixed constant,
handling the valid pixel range (0-255) correctly using cv2.add
(which automatically saturates instead of wrapping around).
"""
import cv2
import numpy as np

img = cv2.imread("input.jpg")

# Brightness constant to add
BRIGHTNESS_CONSTANT = 80

# Create a matrix of the same shape filled with the constant value
brightness_matrix = np.ones(img.shape, dtype="uint8") * BRIGHTNESS_CONSTANT

# cv2.add handles saturation (clips at 255) instead of overflow wrap-around
brightened = cv2.add(img, brightness_matrix)

cv2.imwrite("output.png", brightened)

# Compare a pixel before and after enhancement
sample_y, sample_x = 150, 200
print(f"Pixel at ({sample_y},{sample_x}) BEFORE enhancement:", img[sample_y, sample_x])
print(f"Pixel at ({sample_y},{sample_x}) AFTER  enhancement:", brightened[sample_y, sample_x])
