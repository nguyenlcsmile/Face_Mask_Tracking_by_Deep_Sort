from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from model_utils import define_model, load_model
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from PIL import Image
import numpy as np
import cv2
import os

def realtime_webcam():
	model = define_model()
	model, net = load_model(model)

	#Các tham số tracking
	# Definition of the parameters
	max_cosine_distance = 0.3
	nn_budget = None
	nms_max_overlap = 1.0

	# Deep SORT
	model_filename = 'mars-small128.pb'
	encoder = gdet.create_box_encoder(model_filename, batch_size=1)

	metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
	tracker = Tracker(metric)
	show_detections = True

	cap = cv2.VideoCapture(0)

	while True:
		ret, image = cap.read()

		orig = image.copy()
		(h, w) = image.shape[:2]
		# chuyển image sang blob
		blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

		# face detections
		print("[INFO] computing face detections...")
		net.setInput(blob)
		detections = net.forward()

		boxes = []
		confidences = []
		# lặp qua các detections
		for i in range(0, detections.shape[2]):
			# lấy ra độ tin cậy (xác suất,...) tương ứng của mỗi detection
			confidence = detections[0, 0, i, 2]
			# print(confidence)
			# lọc ra các detections đảm bảo độ tin cậy > ngưỡng tin cậy
			if confidence > 0.5:
				# tính toán (x,y) bounding box
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")

				boxes.append([startX, startY, endX - startX, endY - startY])
				confidences.append(confidence)
				# print(boxes, len(boxes), type(boxes))
				# print(confidences, len(confidences), type(confidences))
				classes = ["face" for i in range(len(boxes))]
				features = encoder(image, boxes)
				detections_tracking = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in 
					zip(np.array(boxes), np.array(confidences), classes, features)]

				# Run non-maxima suppression.
				# boxess = np.array([d.tlwh for d in detections_tracking])
				scores = np.array([d.confidence for d in detections_tracking])
				classes = np.array([d.cls for d in detections_tracking])
				indices = preprocessing.non_max_suppression(np.array([d.tlwh for d in detections_tracking]), nms_max_overlap, scores)
				detections_tracking = [detections_tracking[i] for i in indices]

				# Call the tracker
				tracker.predict()
				tracker.update(detections_tracking)
				try:
					for det in detections_tracking:
						bbox = det.to_tlbr()
						if show_detections and len(classes) > 0:
							# đảm bảo bounding box nằm trong kích thước frame
							(startX, startY) = (max(0, bbox[0]), max(0, bbox[1]))
							(endX, endY) = (min(w - 1, bbox[2]), min(h - 1, bbox[3]))

							# trích ra face ROI, chuyển image từ BGR sang RGB, resize về 224x224 và preprocess
							face = image[int(startY):int(endY), int(startX):int(endX)]
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
							cv2.putText(image, label, (int(startX), int(startY) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
							cv2.rectangle(image, (int(startX), int(startY)), (int(endX), int(endY)), color, 2)

					for track in tracker.tracks:
						print(track)
						if not track.is_confirmed() or track.time_since_update > 1:
							continue
						bbox = track.to_tlbr()
						# print(bbox)
						(startX, startY) = (max(0, bbox[0]), max(0, bbox[1]))
						(endX, endY) = (min(w - 1, bbox[2]), min(h - 1, bbox[3]))
						# adc = "%.2f" % (track.adc * 100) + "%"  # Average detection confidence
						cv2.rectangle(image, (int(startX), int(startY)), (int(endX), int(endY)), (255, 255, 255), 2)
						cv2.putText(image, "ID: " + str(track.track_id), (int(startX), int(endY) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
				except:
					pass
		if cv2.waitKey(10) & 0xFF == ord("q"):
			break
		# show output image
		cv2.imshow("Output", cv2.resize(image, (720, 480)))
		# cleanup
	# When everything is done, release the capture
	cap.release()
	cv2.destroyAllWindows()

