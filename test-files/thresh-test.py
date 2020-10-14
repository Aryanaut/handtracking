import cv2
import dlib
import numpy as np

def nothing(x):
    pass

cv2.namedWindow('frame')
cap = cv2.VideoCapture(0)

cv2.createTrackbar('threshold', 'frame', 0, 255, nothing)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    threshold = cv2.getTrackbarPos('threshold', 'frame')
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('frame', thresh)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()