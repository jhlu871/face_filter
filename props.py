# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 11:15:07 2017

@author: Jason
"""

import cv2
import numpy as np


def remove_background(img):
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    ret,thresh = cv2.threshold(gray,250,255,cv2.THRESH_BINARY_INV)
    
    kernel = np.ones((2,2),np.float32)/4
    smoothed = cv2.filter2D(thresh,-1,kernel)
    b,g,r = cv2.split(img)
    res = cv2.merge((b,g,r,smoothed))
    return res
    
    
if __name__ == '__main__':
    img = cv2.imread('cowboy.png',-1)
    res = remove_background(img)
    
    #cv2.imshow('img',img)
    cv2.imshow('ret',res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()