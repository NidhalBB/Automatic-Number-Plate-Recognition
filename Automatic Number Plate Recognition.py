import cv2
import numpy as np
import imutils
import easyocr

img = cv2.imread('C:/Users/nidhal/Desktop/1.jpg') #read image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#recolored the image

bfilter = cv2.bilateralFilter(gray, 11, 17, 17) #Noise(bruit) reduction
edged = cv2.Canny(bfilter, 30, 200) #Edge detection

keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)#findcontour and pass the edge and cv2.CHAIN_APPROX_SIMPLE 
#to represent the line with only two point
#the cv2.RETR_TREE to shwo the hierarchy order of line
contours = imutils.grab_contours(keypoints) #grab our contour
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10] #sorted contour and get the top 10 descendante

location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)#approximiser tho polygon from our contour
    if len(approx) == 4:
        location = approx
        break

mask = np.zeros(gray.shape, np.uint8)#do mask for the imag to show only the automatic number
new_image = cv2.drawContours(mask, [location], 0,255, -1)
new_image = cv2.bitwise_and(img, img, mask=mask)

(x,y) = np.where(mask==255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray[x1:x2+10, y1:y2+10]
cv2.imshow('i',cropped_image)
reader = easyocr.Reader(['en'])
result = reader.readtext(cropped_image)

text = result[0][-2]
print(text)
font = cv2.FONT_HERSHEY_SIMPLEX
res = cv2.putText(img, text=text, org=(approx[0][0][0], approx[1][0][1]+60), fontFace=font, fontScale=1, color=(0,255,0), thickness=2, lineType=cv2.LINE_AA)
res = cv2.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0),3)

cv2.imshow('image',res)
cv2.waitKey(0) & 0xFF 
cv2.destroyAllWindows()
