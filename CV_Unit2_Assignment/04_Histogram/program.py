"""
Program 4: Histogram Analysis
Reads a grayscale image, computes and plots its intensity histogram,
saves the plot, and prints the intensity value with the highest frequency.
"""
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

# Compute histogram: 256 bins, range 0-255
hist = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()

# Intensity value with highest frequency (the mode)
peak_intensity = int(np.argmax(hist))
peak_count = int(hist[peak_intensity])
print("Intensity value with highest frequency:", peak_intensity)
print("Frequency (pixel count) at that intensity:", peak_count)

# Plot histogram
plt.figure(figsize=(7, 5))
plt.plot(hist, color="black")
plt.title("Grayscale Intensity Histogram")
plt.xlabel("Pixel Intensity (0-255)")
plt.ylabel("Frequency")
plt.axvline(peak_intensity, color="red", linestyle="--",
            label=f"Peak = {peak_intensity}")
plt.legend()
plt.savefig("output.png", bbox_inches="tight")
plt.close()
