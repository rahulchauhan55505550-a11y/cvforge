"""
Program 6: Mean Filtering
Applies a mean (average) filter to an image using two different
kernel sizes, and keeps the final saved output using the larger kernel.
"""
import cv2

img = cv2.imread("input.jpg")

# Try a smaller kernel first
kernel_small = (3, 3)
smoothed_small = cv2.blur(img, kernel_small)
cv2.imwrite("output_kernel3.png", smoothed_small)  # intermediate test result

# Larger kernel -> stronger smoothing (this is our required final output)
kernel_large = (9, 9)
smoothed_large = cv2.blur(img, kernel_large)
cv2.imwrite("output.png", smoothed_large)

print("Applied mean filter with kernel size", kernel_small, "-> output_kernel3.png (test)")
print("Applied mean filter with kernel size", kernel_large, "-> output.png (final required output)")
