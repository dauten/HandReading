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




#build blob detector
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 0
params.maxThreshold = 500


#Basically we don't care how big the blob is or how circular it is.
#Any collection of pixels is a potential character
params.filterByArea = False
params.filterByConvexity = False
params.filterByCircularity = False

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)




# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str,
	help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf",
	help="OpenCV object tracker type")
args = vars(ap.parse_args())

img = cv2.imread('text.png',0)
edges = cv2.Canny(img,100,200)




# Detect blobs.
keypoints = detector.detect(edges)
print(str(keypoints))
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(edges, keypoints, np.array([]), (200,0,200), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("Frame", im_with_keypoints)
# Detect blobs.
keypoints = detector.detect(edges)


# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(edges, keypoints, np.array([]), (200,0,200), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("Frame", im_with_keypoints)

key = cv2.waitKey(0) & 0xFF


# initialize OpenCV's special multi-object tracker
trackers = cv2.MultiTracker_create()

# if a video path was not supplied, grab the reference to the web cam
if not args.get("video", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(1.0)

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])

files = []

# loop over frames from the video stream
while True:
	# grab the current frame, then handle if we are using a
	# VideoStream or VideoCapture object
	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame


	# check to see if we have reached the end of the stream
	if frame is None:
		break

	# resize the frame (so we can process it faster)
	frame = imutils.resize(frame, width=600)
	edges = cv2.Canny(frame,100,200)

	# Detect blobs.
	keypoints = detector.detect(edges)

	# Draw detected blobs as red circles.
	# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
	# the size of the circle corresponds to the size of blob
	im_with_keypoints = cv2.drawKeypoints(edges, keypoints, np.array([]), (200,0,200), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	cv2.imshow("Frame", im_with_keypoints)

	key = cv2.waitKey(1) & 0xFF




# if we are using a webcam, release the pointer
if not args.get("video", False):
	vs.stop()

# otherwise, release the file pointer
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()
