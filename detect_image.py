from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from model_utils import define_model, load_model
import numpy as np
import cv2
import os

def detect_image(path):

	model = define_model()
	model, net = load_model(model)

	# load input image và preprocess

	image = cv2.imread(path)
	orig = image.copy()
	(h, w) = image.shape[:2]

	# chuyển image sang blob
	blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

	# face detections
	print("[INFO] computing face detections...")
	net.setInput(blob)
	detections = net.forward()

	# lặp qua các detections
	for i in range(0, detections.shape[2]):
		# lấy ra độ tin cậy (xác suất,...) tương ứng của mỗi detection
		confidence = detections[0, 0, i, 2]

		# lọc ra các detections đảm bảo độ tin cậy > ngưỡng tin cậy
		if confidence > 0.5:
			# tính toán (x,y) bounding box
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			
			# đảm bảo bounding box nằm trong kích thước frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# trích ra face ROI, chuyển image từ BGR sang RGB, resize về 224x224 và preprocess
			face = image[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)
			face = np.expand_dims(face, axis=0)

			# dùng model đã train để predict mask or no mask
			(mask, withoutMask) = model.predict(face)[0]

			# xác định class label và color để vẽ bounding box và text
			label = "Mask" if mask > withoutMask else "No Mask"
			color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

			# đính thêm thông tin về xác suất(probability) của label
			label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

			# display label và bounding box hình chữ nhật trên output frame
			cv2.putText(image, label, (startX, startY - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
			cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

	# show output image
	cv2.imshow("Output", image)
	cv2.waitKey(0)

