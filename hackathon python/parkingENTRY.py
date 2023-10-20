import cv2
import numpy as np

# Read the video file from Google Drive
cap = cv2.VideoCapture('video.mp4')

count_line_position = 550

min_width_react=80
min_height_react=80
#initialization subtractor
alg = cv2.bgsegm.createBackgroundSubtractorMOG()

def center_handle(x,y,w,h):
    x1=int(w/2)
    y1=int(h/2)
    cx=x+x1
    cy=y+y1
    return cx,cy

detec = []
offset = 6 # allowable error between pixels
count = 0

while True:
  ret,frame1 = cap.read()
  grey = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
  blur = cv2.GaussianBlur(grey,(3,3),5)
  
  #application for each frame
  
  img_sub = alg.apply(blur)
  dilat = cv2.dilate(img_sub,np.ones((5,5)))
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
  dilatada = cv2.morphologyEx(dilat,cv2.MORPH_CLOSE,kernel)
  dilatada = cv2.morphologyEx(dilatada,cv2.MORPH_CLOSE,kernel)
  counter,h = cv2.findContours(dilatada,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  
  
  cv2.line(frame1,(20,count_line_position),(600,count_line_position),(255,127,0),3)
  
  
  for (i,c) in enumerate(counter):
      (x,y,w,h) = cv2.boundingRect(c)
      val_counter = ( w >= min_width_react) and ( h >= min_height_react)
      if not val_counter:
          continue
    
      cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,0,255),1)
      
      center = center_handle(x,y,w,h)
      detec.append(center)
      cv2.circle(frame1,center,4,(0,0,255),-1)
      
      for(x,y) in detec:
          if y<(count_line_position+offset) and y>(count_line_position-offset):
              count+=1
          cv2.line(frame1,(20,count_line_position),(600,count_line_position),(255,110,0),3)
          detec.remove((x,y))
          print("vehicle parked :" +str(count))
  if count<=40:            
    cv2.putText(frame1,"vehicle parked : "+str(count),(50,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,225),2)
  else:
    cv2.putText(frame1,"NO parking available",(50,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,225),2)
          
  # just to show what happens in backend cv2.imshow('Detector',dilatada)
    
    
  
  
  
  
  cv2.imshow('original video',frame1)

  if cv2.waitKey(1) == 13:
    break


cv2.destroyAllWindows()
cap.release()