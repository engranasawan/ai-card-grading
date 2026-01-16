import cv2
import numpy as np

def whiteness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return ((hsv[:,:,2] > 200) & (hsv[:,:,1] < 40)).mean()

def corner_score_from_patches(corners):
    w = max(whiteness(p) for p in corners.values())
    if w < 0.04: return 10
    if w < 0.08: return 8
    if w < 0.15: return 6
    return 4
