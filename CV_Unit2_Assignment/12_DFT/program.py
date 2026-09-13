"""
Program 12: 2D DFT Computation
Reads a grayscale image, computes its 2D DFT using OpenCV, and
shifts the frequency representation so the low-frequency component
is centered. Prints shapes at each stage.
"""
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

# Convert to float32 as required by cv2.dft
img_float32 = np.float32(img)

# Compute 2D DFT (two channels: real and imaginary parts)
dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)

# Shift the zero-frequency (low-frequency) component to the center
dft_shift = np.fft.fftshift(dft)

print("Original image shape:", img.shape)
print("DFT result shape:", dft.shape)
print("Shifted DFT result shape:", dft_shift.shape)

# Save a simple visualization of the centered magnitude as the required output
magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])
magnitude_log = 20 * np.log(magnitude + 1)

plt.figure(figsize=(5, 5))
plt.imshow(magnitude_log, cmap="gray")
plt.title("Centered DFT (log magnitude)")
plt.axis("off")
plt.savefig("output.png", bbox_inches="tight")
plt.close()
