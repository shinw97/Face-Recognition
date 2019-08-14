import numpy as np
import cv2


CASCADE_XML_DIR = 'haarcascade_frontalface_default.xml'

face_cascade = cv2.CascadeClassifier(CASCADE_XML_DIR)

cap = cv2.VideoCapture(0)

i = 1
while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()

	# Our operations on the frame come here
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	

	list_faces = [w * h for (x,y,w,h) in faces]
	if list_faces != []:
		max_area_face_index = list_faces.index(max(list_faces))
		x, y, w, h = faces[max_area_face_index]
		crop_frame = frame[y: y+h, x: x+w]
		crop_frame = cv2.flip(crop_frame, 1)	
		cv2.rectangle(frame,(x, y),(x + w, y + h),(255,0,0),2)

	# for (x,y,w,h) in faces:	
	# 	crop_frame = frame[y - 30: y+h, x: x+w]
	# 	crop_frame = cv2.flip(crop_frame, 1)	
	# 	cv2.rectangle(frame,(x, y - 30),(x + w, y + h),(255,0,0),2)
	
	# Display the resulting frame
	frame = cv2.flip(frame, 1)
	cv2.imshow('frame', frame)

	k = cv2.waitKey(33)
	if k == 112: # key 'p'
		if list_faces == []:
			print('No faces detected!')
			continue
		cv2.imwrite('faces-dataset/true-faces-' + str(i) + '.jpg', crop_frame)
		print('Captured true-faces-' + str(i) + '.jpg')
		i += 1
		continue
	elif k == 113: # key 'q'
		break
	# elif k == 97: # key 'a'
	# 	print(w * h)
	# 	continue
	else:  # normally -1 returned
		continue

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()