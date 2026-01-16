import cv2
import numpy as np

def smooth_1d(x, k):
    k = max(3, k | 1)
    return np.convolve(x, np.ones(k)/k, mode="same")

def compute_centering_final(aligned):
    gray = cv2.cvtColor(aligned, cv2.COLOR_BGR2GRAY)
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
    col = smooth_1d(lap.mean(axis=0), int(0.05 * aligned.shape[1]))
    row = smooth_1d(lap.mean(axis=1), int(0.05 * aligned.shape[0]))

    col /= col.max() + 1e-6
    row /= row.max() + 1e-6

    left = np.argmax(col > 0.3)
    right = aligned.shape[1] - np.argmax(col[::-1] > 0.3)
    top = np.argmax(row > 0.3)
    bottom = aligned.shape[0] - np.argmax(row[::-1] > 0.3)

    if right <= left or bottom <= top:
        return (None, aligned), "failed"

    LR = min(left, aligned.shape[1]-right) / max(left, aligned.shape[1]-right)
    TB = min(top, aligned.shape[0]-bottom) / max(top, aligned.shape[0]-bottom)

    return ({
        "LR_ratio": LR,
        "TB_ratio": TB
    }, aligned), "variance"

def centering_score_from_ratios_strict(LR, TB):
    r = min(LR, TB)
    return 10 if r > 0.95 else \
           9 if r > 0.90 else \
           8 if r > 0.85 else \
           7 if r > 0.80 else \
           6 if r > 0.75 else \
           5 if r > 0.70 else \
           4 if r > 0.60 else \
           3 if r > 0.50 else \
           2 if r > 0.40 else 1

def centering_sensor_mode(LR, TB):
    return centering_score_from_ratios_strict(
        max(LR, 0.55),
        max(TB, 0.55)
    )
