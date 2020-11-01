import cv2
import numpy as np
import dlib

img = cv2.imread('hand.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

_, thr = cv2.threshold(gray, 2, 255, cv2.THRESH_BINARY)
edged = cv2.Canny(gray, 30, 200)

cont, h = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

cv2.imshow('img', edged)
cv2.imwrite('canny.png', edged)
cv2.imwrite('gray.png', gray)
cv2.waitKey(0)

print(str(len(cont)))

cv2.drawContours(img, cont, -1, (0, 255, 0), 3)

cv2.imshow('img2', img)
cv2.imwrite('detected_contours.png', img)
cv2.waitKey(0)
cv2.destroyAllWindows()