# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 19:09:42 2017

@author: Jason
"""

import cv2
import numpy as np
from props import remove_background

cap = cv2.VideoCapture(0)

hat_orig = remove_background(cv2.imread('crown.jpg',-1))

def rescale_props(prop,w):
    return cv2.resize(prop,(w//2,w//2),interpolation=cv2.INTER_CUBIC)
    
def show_hats(img,x,y,w,h):
    hat = rescale_props(hat_orig,w)
    x_offset = 30
    y_offset = 20
    if y-hat.shape[1]+y_offset > 0:
        loc_y = slice(y-hat.shape[1]+y_offset,y+y_offset)
        loc_x = slice(x+x_offset,x+hat.shape[0]+x_offset)
        flat_hat = np.reshape(hat,(-1,4))
        hat_idx = np.tile((flat_hat[:,3]>0)[:,np.newaxis],3)
        flattened = np.where(hat_idx,flat_hat[:,:3], np.reshape(img[loc_y,loc_x],(-1,3)))
        img[loc_y,loc_x] = np.reshape(flattened,hat.shape[:2]+(3,))
    
def show_face_features(img,gray,x,y,w,h):
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h,x:x+w]
    roi_color = img[y:y+h,x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for ex,ey,ew,eh in eyes:
        cv2.circle(roi_color,(ex+ew//2,ey+eh//2),ew//2,(255,255,255),-1)
        cv2.circle(roi_color,(ex+ew//2,ey+eh//2),ew//10,(0,0,0),-1)
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))

face_cascade = cv2.CascadeClassifier('face.xml')
eye_cascade = cv2.CascadeClassifier('eye.xml')

face_on = True
hats_on = False
while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray)
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break
    elif k == 104:
        hats_on = True
        face_on = False
    elif k == 102:
        face_on = True
        hats_on = False
    
#    out.write(frame)
    
    for x,y,w,h in faces:
        #cv2.ellipse(img,(x+w//2,y+h//2),(int(w*.4),h//2),0,0,360,(255,255,255),-1)
       
        if hats_on:
            show_hats(img,x,y,w,h)
        if face_on:
            show_face_features(img,gray,x,y,w,h)
     
    cv2.imshow('img',img)

cap.release()
#out.release()
cv2.destroyAllWindows()