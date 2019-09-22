import numpy as np
import cv2

from keras.models import load_model


def load_model_from_dir(model_dir):
	model = load_model(model_dir)
	return model


def retrieve_image_from_database():
	image_dir = 'database/true-faces-1.jpg'
	true_face = cv2.imread(image_dir)
	true_face = cv2.resize(true_face, (200, 200))
	true_face = normalize(true_face)
	# print(true_face.shape)
	return true_face

def normalize(reshaped_image):
	def norm(reshaped_image):
		reshaped_image = reshaped_image.astype('float')
		reshaped_image[..., 0] -= 103.939
		reshaped_image[..., 1] -= 116.779
		reshaped_image[..., 2] -= 123.68
		return reshaped_image
	return norm(reshaped_image)


def calculate_distance(image_a, model):

	image_a = cv2.resize(image_a, (200, 200))
	image_a = np.expand_dims(image_a, axis=0)
	image_a = normalize(image_a)
	image_a_encoding = model.predict(image_a)
	# print(image_a_encoding.shape)

	image_b = retrieve_image_from_database()
	image_b = np.expand_dims(image_b, axis=0)
	image_b_encoding = model.predict(image_b)
	# print(image_b_encoding.shape)

	# print(image_a_encoding)
	# print(image_b_encoding)

	dist = np.linalg.norm(image_a_encoding - image_b_encoding)
	print(dist)
	return dist

if __name__ == '__main__':
	
	model = load_model_from_dir('vgg16_modified.h5')
	retrieve_image_from_database()

	CASCADE_XML_DIR = 'haarcascade_frontalface_default.xml'

	face_cascade = cv2.CascadeClassifier(CASCADE_XML_DIR)

	cap = cv2.VideoCapture(0)

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
		# for (x,y,w,h) in faces:
			crop_frame = frame[y: y+h, x: x+w]
			crop_frame = cv2.flip(crop_frame, 1)
			if crop_frame.shape[0] >= 200 and crop_frame.shape[1] >= 200:
				cv2.rectangle(frame,(x, y),(x + w, y + h),(255,0,0),2)
				# print(crop_frame.shape)
				dist = calculate_distance(crop_frame, model)
				if dist < 300:
					print('Verified!')
					break

		# Display the resulting frame
		frame = cv2.flip(frame, 1)
		cv2.imshow('frame', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	# When everything done, release the capture
	cap.release()
	cv2.destroyAllWindows()