"""
Program 13: Magnitude Spectrum
Computes the DFT and centered, log-scaled magnitude spectrum of a
grayscale image and saves it for visualization.
"""
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
img_float32 = np.float32(img)

dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Magnitude = sqrt(real^2 + imaginary^2)
magnitude = cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])

# Log scaling to compress the large dynamic range into a viewable range
magnitude_spectrum = 20 * np.log(magnitude + 1)

plt.figure(figsize=(5, 5))
plt.imshow(magnitude_spectrum, cmap="gray")
plt.title("Magnitude Spectrum")
plt.axis("off")
plt.savefig("output.png", bbox_inches="tight")
plt.close()

print("Magnitude spectrum computed and saved.")
print("Magnitude spectrum shape:", magnitude_spectrum.shape)
