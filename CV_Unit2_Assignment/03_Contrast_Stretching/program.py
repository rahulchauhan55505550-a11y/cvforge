"""
Program 3: Contrast Stretching
Reads a low-contrast grayscale image, finds its min and max intensity,
and linearly stretches the intensity range to 0-255.
(Manual contrast stretching -- NOT histogram equalization.)
"""
import cv2
import numpy as np

# Read as grayscale directly
img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

min_val = np.min(img)
max_val = np.max(img)
print("Minimum intensity in input image:", min_val)
print("Maximum intensity in input image:", max_val)

# Linear contrast stretching formula:
# output = (input - min) * (255 / (max - min))
stretched = (img.astype(np.float32) - min_val) * (255.0 / (max_val - min_val))
stretched = np.clip(stretched, 0, 255).astype(np.uint8)

print("New minimum intensity after stretching:", np.min(stretched))
print("New maximum intensity after stretching:", np.max(stretched))

cv2.imwrite("output.png", stretched)
