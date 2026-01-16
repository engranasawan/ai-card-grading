import cv2
import numpy as np

def whiteness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return ((hsv[:,:,2] > 200) & (hsv[:,:,1] < 40)).mean()

def edge_score_from_patches(edges):
    w = max(whiteness(p) for p in edges.values())
    if w < 0.05: return 10
    if w < 0.10: return 8
    if w < 0.18: return 6
    return 4
