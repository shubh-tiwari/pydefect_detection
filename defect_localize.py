import cv2
import numpy as np
from skimage import io
from matplotlib import pyplot as plt

def hist_eq(img):
    """Histogram equalization"""
    lab_img= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    #Splitting the LAB image to L, A and B channels, respectively
    l, a, b = cv2.split(lab_img)

    #Apply histogram equalization to the L channel
    equ = cv2.equalizeHist(l)

    #Combine the Hist. equalized L-channel back with A and B channels
    updated_lab_img1 = cv2.merge((equ,a,b))

    #Convert LAB image back to color (RGB)
    hist_eq_img = cv2.cvtColor(updated_lab_img1, cv2.COLOR_LAB2BGR)

    return hist_eq_img

def clah_eq(img):
    """Contrast limited adaptive histogram equalization"""
    lab_img= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    #Splitting the LAB image to L, A and B channels, respectively
    l, a, b = cv2.split(lab_img)

    #Apply CLAHE to L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    clahe_img = clahe.apply(l)
    #plt.hist(clahe_img.flat, bins=100, range=(0,255))

    #Combine the CLAHE enhanced L-channel back with A and B channels
    updated_lab_img2 = cv2.merge((clahe_img,a,b))

    #Convert LAB image back to color (RGB)
    CLAHE_img = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)

    return CLAHE_img

def extract_decoloration(img):
    """Function to extract decoloration zone"""
    #Enhancing image using CLAHE technique
    clahe_img = clah_eq(img)

    #Converting BGR to gray image
    clahe_gray = cv2.cvtColor(clahe_img, cv2.COLOR_BGR2GRAY)

    #Smoothing the image
    clahe_blur = cv2.medianBlur(clahe_gray,5)

    #Binarizing the image using user defined thresholding parameter
    _,clahe_thresh = cv2.threshold(clahe_blur,190,255,cv2.THRESH_BINARY)

    #Performing dilation operation on the binarized image
    kernel = np.ones((19,3),np.uint8)
    clahe_dilate = cv2.dilate(clahe_thresh, kernel,iterations = 1)

    #Detecting decolored zone using contours
    contours, _ = cv2.findContours(clahe_dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        rect = cv2.boundingRect(c)
        x,y,w,h = rect
        if h*w > 3000:
            img = cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 5)

    return img