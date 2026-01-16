import cv2
import numpy as np
from pathlib import Path

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_warp(image, pts, out_w=1200, out_h=1680):
    rect = order_points(pts.astype("float32"))
    dst = np.array([
        [0, 0],
        [out_w - 1, 0],
        [out_w - 1, out_h - 1],
        [0, out_h - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (out_w, out_h))

def detect_card_quad_robust(image):
    h, w = image.shape[:2]
    scale = 1000 / w
    resized = cv2.resize(image, (1000, int(h * scale)))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    strategies = []

    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 5
    )
    strategies.append(("adaptive", th))

    edges = cv2.Canny(gray, 50, 150)
    strategies.append(("canny", edges))

    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (0, 0, 160), (180, 60, 255))
    strategies.append(("light_mask", mask))

    for name, bin_img in strategies:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel, iterations=2)

        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        for c in cnts[:10]:
            if cv2.contourArea(c) < 0.15 * resized.size:
                continue
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                return (approx.reshape(4, 2) / scale).astype(int), name

    return None, None

def align_card_robust(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None, None

    quad, strategy = detect_card_quad_robust(img)
    if quad is None:
        return None, img, None

    warped = four_point_warp(img, quad)
    return warped, img, strategy
