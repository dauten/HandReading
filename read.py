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
from interpret import interpreter
import PIL

# initialize OpenCV's special multi-object tracker
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
	help="path to input image file")
args = vars(ap.parse_args())

# if a video path was not supplied, grab the reference to the web cam
if not args.get("image", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(1.0)
else:
	vs = cv2.VideoCapture(args["image"])



# initialize OpenCV's special multi-object tracker
trackers = cv2.MultiTracker_create()

model = interpreter()



# loop over frames from the video stream
while True:

	if not args.get("image", False):
		frame = vs.read()
	else:
		frame = cv2.imread(args["image"])

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
					if roi[a, b][z] < 120:
						roi[a, b][z] = 0
					else:
						roi[a, b][z] = 255


		cv2.imwrite(str(n)+".png", roi[y+2:y+h-1, x+2:x+w-1])

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
	elif key == ord("e"):
		print("Reading images:")
		for n, box in enumerate(boxes):

			fname = str(n)+".png"
			print("Evaluating "+fname)
			img = PIL.Image.open(fname).convert("L")
			print(model.eval(img))




# if we are using a webcam, release the pointer
if not args.get("image", False):
	vs.stop()
# otherwise, release the file pointer
else:
	vs.release()



# close all windows
cv2.destroyAllWindows()
