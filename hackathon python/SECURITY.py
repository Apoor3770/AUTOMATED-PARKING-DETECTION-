import cv2
import pytesseract
image = cv2.imread('license_car.jpg')
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
canny=cv2.Canny(gray,170,200)
contours,new=cv2.findContours(canny.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
contours=sorted(contours,key=cv2.contourArea,reverse = True)[:30]
contours_with_license_plate=None
license_plate=None
x=None
y=None
w=None
h=None
for contour in contours:
        perimeter=cv2.arcLength(contour,True)
        approx=cv2.approxPolyDP(contour,0.01*perimeter,True)
        if len(approx)==4:
            contours_with_license_plate=approx
            x,y,w,h=cv2.boundingRect(contour)
            license_plate=gray[y:y+h,x:x+w]
            break

license_plate=cv2.bilateralFilter(license_plate,11,17,17)
(thresh,license_plate)=cv2.threshold(license_plate,150,180,cv2.THRESH_BINARY)
text=pytesseract.image_to_string(license_plate)
image=cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),3)
image=cv2.putText(image,text,(x-100,y-50),cv2.FONT_HERSHEY_SIMPLEX,3,(0,225,0),6,cv2.LINE_AA)
print("license plate:",text)
cv2.imshow("license plate Detector",image)
cv2.waitKey(0)