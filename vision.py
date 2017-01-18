# -*- coding: utf-8 -*-


import cv2
import dlib
import numpy as np
from props import remove_background

cap = cv2.VideoCapture(0)

hat_orig = remove_background(cv2.imread('crown.jpg',-1))

def rescale_props(prop,w):
    return cv2.resize(prop,(w//2,w//2),interpolation=cv2.INTER_CUBIC)
    
def show_hats(img,x,y,w,h):
    
    hat = rescale_props(hat_orig,w)
    x_offset = 50
    y_offset = 20
    if y-hat.shape[1]+y_offset > 0:
        mask = hat[:,:,3]
        mask_inv = cv2.bitwise_not(mask)
        hat = hat[:,:,:3]
        loc_y = slice(y-hat.shape[1]+y_offset,y+y_offset)
        loc_x = slice(x+x_offset,x+hat.shape[0]+x_offset)
        roi = img[loc_y,loc_x]
        roi_bg = cv2.bitwise_and(roi,roi,mask=mask_inv)
        roi_fg = cv2.bitwise_and(hat,hat,mask=mask)
        dst = cv2.add(roi_bg,roi_fg)
        
        """ Seamless Clone """
#        w1,h1 = hat.shape[:2]
#        dst = cv2.seamlessClone(hat,img[loc_y,loc_x],mask,(w1//2,h1//2),cv2.MIXED_CLONE)

        """ Pure Numpy """
#        flat_hat = np.reshape(hat,(-1,4))
#        hat_idx = np.tile((flat_hat[:,3]>0)[:,np.newaxis],3)
#        flattened = np.where(hat_idx,flat_hat[:,:3], np.reshape(img[loc_y,loc_x],(-1,3)))
        img[loc_y,loc_x]= dst
    
def show_face_features(img,gray,x,y,w,h):
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h,x:x+w]
    roi_color = img[y:y+h,x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for ex,ey,ew,eh in eyes:
        cv2.circle(roi_color,(ex+ew//2,ey+eh//2),ew//2,(255,255,255),-1)
        cv2.circle(roi_color,(ex+ew//2,ey+eh//2),ew//10,(0,0,0),-1)
        
def convert_dlib_results_to_cv(face):
    x = face.left()
    y = face.top()
    w = face.right()-x
    h = face.bottom() - y
    return x,y,w,h
    
    
if __name__=='__main__':
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))

    use_dlib = False
    if use_dlib:
        detector = dlib.get_frontal_face_detector()
    else:
        face_cascade = cv2.CascadeClassifier('face.xml')
        eye_cascade = cv2.CascadeClassifier('eye.xml')
    
    face_on = False
    hats_on = False
    face_swap = False
    while True:
        ret, img = cap.read()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        if use_dlib:
            faces = detector(img,1)
        else:
            faces = face_cascade.detectMultiScale(gray)
        
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k == 104:
            hats_on = True
            face_on = False
            face_swap = False
        elif k == 102:
            face_on = True
            hats_on = False
            face_swap = False
        elif k == 115:
            face_on = False
            hats_on = False
            face_swap = True
        
#        out.write(frame)

        face_array = []
        for face in faces:
            if use_dlib:
                x,y,w,h = convert_dlib_results_to_cv(face)
            else:
                x,y,w,h = face
            
            roi = img[y:y+h,x:x+w]
            roi_mask = np.zeros(roi.shape[:2],np.uint8)
            cv2.ellipse(roi_mask,(w//2,h//2),(int(w*.32),int(h*.45)),0,0,360,(255,255,255),-1)
            roi_mask_inv = cv2.bitwise_not(roi_mask)
            roi_mask_fg = cv2.bitwise_and(roi,roi,mask=roi_mask)
            roi_mask_bg = cv2.bitwise_and(roi,roi,mask=roi_mask_inv)
                      
#            img[y:y+h,x:x+w]=roi_mask_fg
            face_array.append((roi_mask_fg,roi_mask_bg))
            
            
            if hats_on:
                show_hats(img,x,y,w,h)
            if face_on:
                show_face_features(img,gray,x,y,w,h)
                
        if len(face_array)==2 and face_swap:
            for i,(x,y,w,h) in enumerate(faces):
                if i == 0:
                    (fg,_) = face_array[1]
                    (_,bg) = face_array[0]
                else:
                    (fg,_) = face_array[0]
                    (_,bg) = face_array[1]
                img[y:y+h,x:x+w]=cv2.add(bg,cv2.resize(fg,(w,h)))
         
        cv2.imshow('img',img)
    
    cap.release()
    #out.release()
    cv2.destroyAllWindows()