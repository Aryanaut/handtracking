from PIL import Image
import numpy as np
import cv2


test = np.zeros((300, 300, 3), np.uint8)

cv2.circle(test, (150, 150), 3, (255, 255, 0), -1)
img = Image.fromarray(test, 'RGB')
img.save('test.png')

cv2.imshow('test', test)
cv2.waitKey(0)
cv2.destroyAllWindows()