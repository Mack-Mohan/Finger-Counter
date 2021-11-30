import cv2
import numpy as np
from sklearn.metrics import pairwise 

#Global variables

background = None

#ROI

roi_top = 50
roi_bottom = 300
roi_right = 250
roi_left = 600


#Alpha for accumulating images

alpha = 0.5

#average background value

def running_bg(frame) :
    global background, alpha
    if background is None:
        background = frame.copy().astype('float')
        
    cv2.accumulateWeighted(frame, background, alpha)
    
#segmenting the hand
def segmenting(frame) :
    diff = cv2.absdiff(background.astype('uint8'), frame)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    image, contours, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None
    else:
        hand = max(contours, key = cv2.contourArea)
        return (hand, thresh)
        
        
def fingers(hand, thresh):
    conv_hull = cv2.convexHull(hand)
    
    count = 0
    top    = tuple(conv_hull[conv_hull[:, :, 1].argmin()][0])
    bottom = tuple(conv_hull[conv_hull[:, :, 1].argmax()][0])
    left   = tuple(conv_hull[conv_hull[:, :, 0].argmin()][0])
    right  = tuple(conv_hull[conv_hull[:, :, 0].argmax()][0])
    
    #center
    cx = (left[0] + right[0]) //2
    cy = (top[1] + bottom[1]) //2
    
    distance = pairwise.euclidean_distances([(cx,cy)], Y = [left, right, top, bottom])[0]
    max_d = distance.max()
    
    radius = int(0.8*max_d)
    circumference = 2*np.pi*radius
    
    circular = np.zeros(thresh.shape[:2], dtype="uint8")
    
    cv2.circle(circular, (cx,cy), radius, 255, 10)
    cv2.imshow('frame1',circular)
    circular = cv2.bitwise_and(thresh, thresh, mask = circular)
    cv2.imshow('frame2',circular)
    
    _, contour, _ = cv2.findContours(circular.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    for i in contour:
        (x, y, w, h) = cv2.boundingRect(i)
        
        if ((cy + cy*0.25) > (y + h)) and ((circumference*0.25) > i.shape[0]):
            count += 1
            
    return count
    
    
