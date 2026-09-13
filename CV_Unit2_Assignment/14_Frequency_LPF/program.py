"""
Program 14: Frequency-Domain Low-Pass Filtering
Creates a circular low-pass mask that preserves the central
low-frequency region and suppresses higher frequencies, then
reconstructs the filtered image via inverse DFT.
"""
import cv2
import numpy as np

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2

img_float32 = np.float32(img)
dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)

# Circular low-pass mask: 1 inside radius (keep low frequencies),
# 0 outside (suppress high frequencies)
radius = 40
mask = np.zeros((rows, cols, 2), np.uint8)
y, x = np.ogrid[:rows, :cols]
mask_area = (x - ccol) ** 2 + (y - crow) ** 2 <= radius ** 2
mask[mask_area] = 1

# Apply mask to the shifted DFT
fshift_filtered = dft_shift * mask

# Inverse shift and inverse DFT to reconstruct the spatial-domain image
f_ishift = np.fft.ifftshift(fshift_filtered)
img_back = cv2.idft(f_ishift)
img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

# Normalize to 0-255 for saving
img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
img_back = np.uint8(img_back)

cv2.imwrite("output.png", img_back)
print("Low-pass filtering complete. Radius used:", radius)
