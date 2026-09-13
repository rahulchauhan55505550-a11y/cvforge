"""
Program 5: Histogram Equalization with Before/After Comparison
Performs histogram equalization on a grayscale image, saves the
equalized image, and saves a comparison plot of the histograms
before and after equalization.
"""
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

img = cv2.imread("input.jpg", cv2.IMREAD_GRAYSCALE)

# Perform histogram equalization
equalized = cv2.equalizeHist(img)

# Save equalized image
cv2.imwrite("output.png", equalized)

# Compute histograms before and after
hist_before = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
hist_after = cv2.calcHist([equalized], [0], None, [256], [0, 256]).flatten()

# Save comparison plot
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(hist_before, color="blue")
plt.title("Histogram Before Equalization")
plt.xlabel("Intensity")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.plot(hist_after, color="green")
plt.title("Histogram After Equalization")
plt.xlabel("Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("histogram_comparison.png", bbox_inches="tight")
plt.close()

print("Histogram equalization complete.")
print("Original image shape:", img.shape)
print("Equalized image shape:", equalized.shape)
