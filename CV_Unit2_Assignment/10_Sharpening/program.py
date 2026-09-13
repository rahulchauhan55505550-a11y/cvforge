"""
Program 10: Image Sharpening Using a Custom Kernel
Creates a sharpening kernel and applies it with cv2.filter2D().
"""
import cv2
import numpy as np

img = cv2.imread("input.jpg")

# Custom sharpening kernel:
# The center weight (9) boosts the current pixel while the four
# neighboring -1 weights subtract surrounding pixel values, which
# amplifies local intensity differences (edges) -> sharper look.
# The weights sum to 1, so overall image brightness is preserved.
sharpening_kernel = np.array([
    [ 0, -1,  0],
    [-1,  9, -1],
    [ 0, -1,  0]
])

sharpened = cv2.filter2D(img, -1, sharpening_kernel)

cv2.imwrite("output.png", sharpened)
print("Sharpening kernel used:\n", sharpening_kernel)
