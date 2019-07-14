import numpy as np
import cv2

CASCADE_XML_DIR = 'haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier(CASCADE_XML_DIR)

cap = cv2.VideoCapture(0)

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		cv2.rectangle(frame,(x, y - 30),(x + w, y + h + 20),(255,0,0),2)
	# Display the resulting frame
	frame = cv2.flip(frame, 1)
	cv2.imshow('frame', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()