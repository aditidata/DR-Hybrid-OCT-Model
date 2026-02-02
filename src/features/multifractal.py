import numpy as np
import cv2
from scipy.stats import linregress


def box_count(img, box_size):
    h, w = img.shape
    count = 0
    for i in range(0, h, box_size):
        for j in range(0, w, box_size):
            if np.any(img[i:i+box_size, j:j+box_size]):
                count += 1
    return count


def compute_fractal_dimension(img):
    img = cv2.resize(img, (256, 256))
    img = img > img.mean()

    sizes = np.array([2, 4, 8, 16, 32])
    counts = []

    for size in sizes:
        counts.append(box_count(img, size))

    log_sizes = np.log(sizes)
    log_counts = np.log(counts)

    slope, _, _, _, _ = linregress(log_sizes, log_counts)
    return -slope


def extract_multifractal_features(image):
    """
    Input: RGB image (H,W,3)
    Output: feature vector
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    fd = compute_fractal_dimension(gray)
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)

    return np.array([fd, mean_intensity, std_intensity])
