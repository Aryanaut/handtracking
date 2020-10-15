import cv2
import numpy as np
import dlib
from pynput.mouse import *

'''Program to implement full mouse control'''

def nothing(x):
    pass

def clickPos(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        test[:] = 0

# initialising variables
mouse = Controller()
cap = cv2.VideoCapture(0)
cv2.namedWindow('frame')
cv2.namedWindow('drawingWin')
cv2.createTrackbar('Threshold', 'frame', 102, 255, nothing)
drawingWin = np.zeros((1080, 1920, 3), np.uint8)

listofCoords = [(960, 540)]
count = 1
cv2.setMouseCallback('drawingWin', clickPos)

# getting ratios of screen and camera feed
def getPos():
    h, w = (480, 640)
    h1, w1 = (1080, 1920)
    centerPos = (320, 240)

    listofCoords = [(960, 540)]
    count = 1
    ratioX = (w1/centerPos[0])
    ratioY = (h1/centerPos[1])
    return(ratioX, ratioY)

# main function
def main():
    while True:
        # reading the camera
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.flip(frame, 0)
        
        # image processing
        threshold = cv2.getTrackbarPos('Threshold', 'frame')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        _, thr = cv2.threshold(thr, threshold, 255, cv2.THRESH_BINARY)

        edgedThr = cv2.Canny(gray, threshold, 200)
        thr = cv2.GaussianBlur(thr, (5, 5), 0)
        thr = cv2.dilate(thr, None, iterations=2)
        cont, h = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(cont) > 0:
            cv2.drawContours(frame, cont, -1, (0, 0, 255), 3)
            largestCont = max(cont, key = cv2.contourArea)

        cv2.imshow('thr', thr)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# run the program
if __name__ == '__main__':
    main()

# cleanup and exit
cap.release()
cv2.destroyAllWindows()