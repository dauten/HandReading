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


params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200


# Filter by Area.
params.filterByArea = True
params.minArea = 120
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.5




# loop over frames from the video stream
while True:
	# grab the current frame, then handle if we are using a
	# VideoStream or VideoCapture object
	frame = vs.read()

	# check to see if we have reached the end of the stream
	if frame is None:
		break

	# resize the frame (so we can process it faster)
	frame = imutils.resize(frame, width=300)

	(success, boxes) = trackers.update(frame)
	# loop over the bounding boxes and draw then on the frame
	for n, box in enumerate(boxes):
		(x, y, w, h) = [int(v) for v in box]
		roi = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

		#roi is a matrix of colors (3-tuples)
		#that must be converted into matrix of intensities (float or int values)
		#and then all values belo a given threshhold may be dropped?

		for a in range(y, y+h):
			for b in range(x, x+w):
				roi[a, b] = (sum(roi[a, b])/3)
				for z in range(0,3):
					if roi[a, b][z] < 80:
						roi[a, b][z] = 0
					else:
						roi[a, b][z] = 255


		cv2.imwrite(str(n)+".png", roi[y+2:y+h-1, x+2:x+w-1])


	detector = cv2.SimpleBlobDetector_create(params)


	# Detect blobs.
	keypoints = detector.detect(frame)

	# Draw detected blobs as red circles.
	# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
	# the size of the circle corresponds to the size of blob

	frame = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	cv2.imshow("Frame", frame)

	key = cv2.waitKey(1) & 0xFF

	if key == ord("s"):

		# select the bounding box of the object we want to track (make
		# sure you press ENTER or SPACE after selecting the ROI)
		box = cv2.selectROI("Frame", frame, fromCenter=False,
			showCrosshair=True)

		# create a new object tracker for the bounding box and add it
		# to our multi-object tracker
		# other than CSRT options include KCF, Boosting, MIL, TLD, MedianFlow, and MOSSE
		trackers.add(cv2.TrackerKCF_create() , frame, box)
	elif key == ord("q"):
		break

vs.stop()


# close all windows
cv2.destroyAllWindows()
