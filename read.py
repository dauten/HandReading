'''
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('handwriting.jpg',0)
edges = cv2.Canny(img,100,200)

plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()


'''

from imutils.video import VideoStream
import argparse
import imutils
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt
import glob
from imutils.object_detection import non_max_suppression


# initialize OpenCV's special multi-object tracker

print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(1.0)


# initialize OpenCV's special multi-object tracker
trackers = cv2.MultiTracker_create()

# loop over frames from the video stream
while True:
	# grab the current frame, then handle if we are using a
	# VideoStream or VideoCapture object
	frame = vs.read()

	# check to see if we have reached the end of the stream
	if frame is None:
		break

	# resize the frame (so we can process it faster)
	frame = imutils.resize(frame, width=600)

	(success, boxes) = trackers.update(frame)
	# loop over the bounding boxes and draw then on the frame
	for n, box in enumerate(boxes):
		(x, y, w, h) = [int(v) for v in box]
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	cv2.imshow("Frame", frame)

	key = cv2.waitKey(1) & 0xFF
	if key == ord("s"):

		# select the bounding box of the object we want to track (make
		# sure you press ENTER or SPACE after selecting the ROI)
		box = cv2.selectROI("Frame", frame, fromCenter=False,
			showCrosshair=True)

		# create a new object tracker for the bounding box and add it
		# to our multi-object tracker

		trackers.add(cv2.TrackerCSRT_create() , frame, box)

vs.stop()


# close all windows
cv2.destroyAllWindows()
