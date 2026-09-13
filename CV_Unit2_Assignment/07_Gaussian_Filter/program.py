"""
Program 7: Gaussian Smoothing
Applies Gaussian smoothing to a noisy image and saves the result.
"""
import cv2

img = cv2.imread("input.jpg")

# Kernel size (7,7) chosen: it is odd (required for a symmetric center pixel)
# and large enough to noticeably reduce the Gaussian noise present in the
# input image, while still preserving the main scene structure/edges
# (a much larger kernel like 15x15 would over-blur the image).
kernel_size = (7, 7)
sigma = 0  # sigma=0 -> OpenCV automatically computes it from the kernel size

smoothed = cv2.GaussianBlur(img, kernel_size, sigma)

cv2.imwrite("output.png", smoothed)
print("Applied Gaussian smoothing with kernel size", kernel_size)
