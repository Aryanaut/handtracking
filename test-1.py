import cv2
import dlib
import numpy as np

'''
Pointing camera at my white desk
Trying to detect my hand through contour method
Convex hull also drawn
'''
def nothing(x):
    pass

cap = cv2.VideoCapture(0)
cv2.namedWindow('frame')
cv2.createTrackbar('threshold', 'frame', 102, 255, nothing)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    threshold = cv2.getTrackbarPos('threshold', 'frame')
    _, thr = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    edgedThr = cv2.Canny(gray, threshold, 200)
    edgedThr = cv2.GaussianBlur(thr, (5, 5), 0)
    edgedThr = cv2.dilate(edgedThr, None, iterations=2)
    cont, h = cv2.findContours(edgedThr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(cont) > 0:
        cv2.drawContours(frame, cont, -1, (0, 0, 255), 3)
        # print('detected')
        c = max(cont, key = cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        hull = []
        for i in range(len(cont)):
            hull.append(cv2.convexHull(cont[i], False))

            cv2.drawContours(frame, hull, i, (0, 255, 0), 3)

        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
        pass
    cv2.imshow('edged', edgedThr)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

