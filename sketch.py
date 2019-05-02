import cv2
import numpy as np
from interpret import interpreter
import PIL


drawing = False # true if mouse is pressed
blank = True
# initialize OpenCV's special multi-object tracker
trackers = cv2.MultiTracker_create()

model = interpreter()
ix,iy = -1,-1
# mouse callback function
def draw(event,x,y,flags,param):
    global ix,iy,drawing,blank
    if event == cv2.EVENT_LBUTTONDOWN:
        blank = False
        drawing = True
        ix,iy = x,y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
                cv2.circle(img,(x,y),10,(0,0,0),-1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.circle(img,(x,y),10,(0,0,0),-1)


img = np.full((512, 512, 3), 255, np.uint8)
cv2.namedWindow('Canvas')
cv2.setMouseCallback('Canvas',draw)
while(1):
    cv2.imshow('Canvas',img)
    k = cv2.waitKey(1) & 0xFF


    if k == ord('e') and not blank:
        cv2.imwrite("temp.png", img)
        thumb = PIL.Image.open("temp.png").convert("L")
        print(model.eval(thumb))
    if k == ord('c'):
        img = np.full((512, 512, 3), 255, np.uint8)
        blank = True
    elif k == 'q':
        break

cv2.destroyAllWindows()
