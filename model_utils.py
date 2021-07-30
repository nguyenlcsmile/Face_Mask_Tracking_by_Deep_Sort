from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import os 
import cv2 

def define_model():
	# load MobileNetV2 network cho fine-tuning (bỏ đi head FC layer)
	baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
	# baseModel.summary()

	# xây phần head của model (fine-tuning)
	headModel = baseModel.output
	headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
	headModel = Flatten(name="flatten")(headModel)
	headModel = Dense(128, activation="relu")(headModel)
	headModel = Dropout(0.5)(headModel)
	headModel = Dense(2, activation="softmax")(headModel)

	# đặt phần head vừa xây vào đầu của model load
	model = Model(inputs=baseModel.input, outputs=headModel)
	
	for layer in baseModel.layers:
		layer.trainable = False

	return model

def load_model(model):
	# load face detector model từ thư mục
	print("[INFO] loading face detector model...")
	prototxtPath = os.path.join("face_detector/deploy.prototxt")
	weightsPath = os.path.join("face_detector/res10_300x300_ssd_iter_140000.caffemodel")
	net = cv2.dnn.readNet(prototxtPath, weightsPath)

	# load face mask detector model đã train
	print("[INFO] loading face mask detector model...")
	model.load_weights("face_mask.h5")

	return model, net
