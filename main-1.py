import cv2
import dlib
import numpy as np
import pyautogui
import mouse


pyautogui.FAILSAFE = False
cv2.namedWindow('test')
test = np.zeros((1080, 1920, 3), np.uint8)
'''
Pointing camera at my white desk
Trying to detect my hand through contour method
Convex hull also drawn
'''
def nothing(x):
    pass

def clickPos(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        test[:] = 0

cap = cv2.VideoCapture(0)
cv2.namedWindow('frame')
cv2.createTrackbar('threshold', 'frame', 102, 255, nothing)
h, w = (480, 640)
h1, w1 = (1080, 1920)
centerPos = (320, 240)

listofCoords = [(960, 540)]
count = 1
ratioX = (w1/centerPos[0])
ratioY = (h1/centerPos[1])
cv2.setMouseCallback('test', clickPos)
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    frame = cv2.flip(frame, 0)
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
        contL= max(cont, key = cv2.contourArea)
        x, y, w, h = cv2.boundingRect(contL)
        hull = []
        hull_drawn = cv2.convexHull(contL, False)
        arCont = cv2.contourArea(contL)
        if arCont >= 5000:
            for i in range(len(cont)):
                hull.append(cv2.convexHull(contL, returnPoints=False))
                defects = cv2.convexityDefects(contL, hull[i])
                # print(len(hull[i]))
                if defects is not None:
                    cnt = 0
                    for defect in range(defects.shape[0]):
                        # cv2.line(frame, start, end, [0, 255, 0], 2)
                        topmost = tuple(contL[contL[:,:,1].argmin()][0])
                        bottommost = tuple(contL[contL[:,:,1].argmax()][0])
                        cv2.circle(frame, topmost, 5, (255, 0, 0), -1)
                        x1 = topmost[0]
                        y1 = topmost[1]
                        px=x1*ratioX/2
                        py=y1*ratioY/1.5
                        coords = (int(px), int(py))
                        listofCoords.append(coords)
                        cv2.line(test, listofCoords[count-1], listofCoords[count], (255, 100, 0), 3)
                        #cv2.circle(test, coords, 3, (0, 255, 0), -1)
                        count += 1
                        
                else:
                    pass
        else:
            cv2.putText(frame, 'waiting for hand...', (0, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        # ctr = np.array(hull[0]).reshape((-1, 1, 2)).astype(np.int32)
        cv2.drawContours(frame, [hull_drawn], -1, (0, 255, 0), 3)

        # cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    else:
        pass
    cv2.imshow('edged', edgedThr)
    cv2.imshow('frame', frame)
    cv2.imshow('test', test)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

