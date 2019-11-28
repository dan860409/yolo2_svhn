# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import warnings 
warnings.filterwarnings("ignore")

import numpy as np
import cv2
from yolo.backend.utils.box import draw_scaled_boxes
import yolo

from yolo.frontend import create_yolo

# 1. create yolo instance
yolo_detector = create_yolo("ResNet50", ["10", "1", "2", "3", "4", "5", "6", "7", "8", "9"], 416)

# 2. load pretrained weighted file
# Pretrained weight file is at https://drive.google.com/drive/folders/1Lg3eAPC39G9GwVTCH3XzF73Eok-N-dER

DEFAULT_WEIGHT_FILE = os.path.join(yolo.PROJECT_ROOT, "weight/weights.h5")
yolo_detector.load_weights(DEFAULT_WEIGHT_FILE)


# 3. predict testing images
DEFAULT_IMAGE_FOLDER = os.path.join(yolo.PROJECT_ROOT, "data_svhn", "test")
THRESHOLD = 0.25
result_list = []

def bbox_format(boxes):
    '''
        bbox = [[y1, x1, y2, x2], [y1, x1, y2, x2], ...]
    '''
    boxes[:, [0,1]] = boxes[:, [1,0]]
    boxes[:, [2,3]] = boxes[:, [3,2]]
    return boxes

for i in range(1, 13069): # total 13069
	# read image
	img_file = os.path.join(DEFAULT_IMAGE_FOLDER, f"{i}.png")
	img = cv2.imread(img_file)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	print(f"{i}.png", img.shape)

	# three size of images
	crop_imgs = []
	crop_imgs.append(img)
	# crop_imgs.append(sharpen_img(img))
	crop_imgs.append(img[:, int(img.shape[1]*0.2):int(img.shape[1]*0.8), :])
	crop_imgs.append(img[int(img.shape[0]*0.15):int(img.shape[0]*0.85), int(img.shape[1]*0.2):int(img.shape[1]*0.8), :])
	# crop_imgs.append(sharpen_img(img[int(img.shape[0]*0.15):int(img.shape[0]*0.85), int(img.shape[1]*0.2):int(img.shape[1]*0.8), :]))
	# crop_imgs.append(img[int(img.shape[0]*0.25):int(img.shape[0]*0.75), int(img.shape[1]*0.3):int(img.shape[1]*0.7), :])

	# choose best cropping
	best_boxes, best_probs = [], []
	best_avg_confidence = -999
	best_crop_img = img
	best_crop_img_idx = 0

	for crop_img in crop_imgs:
		boxes, probs = yolo_detector.predict(crop_img, THRESHOLD)
		if len(boxes) > 0:
			avg_confidence = np.amax(np.array(probs), axis=1).mean()
			if avg_confidence > best_avg_confidence:
				best_avg_confidence = avg_confidence
				best_boxes = boxes
				best_probs = probs
				best_crop_img = crop_img

	# match boxes coordinate to original image
	if crop_imgs.index(best_crop_img) == 0:
		pass
	elif crop_imgs.index(best_crop_img) == 1:
		best_boxes[:,0] = best_boxes[:,0] + int(img.shape[1]*0.2)
		best_boxes[:,2] = best_boxes[:,2] + int(img.shape[1]*0.2)
	elif crop_imgs.index(best_crop_img) == 2:
		best_boxes[:,0] = best_boxes[:,0] + int(img.shape[1]*0.2)
		best_boxes[:,1] = best_boxes[:,1] + int(img.shape[0]*0.15)
		best_boxes[:,2] = best_boxes[:,2] + int(img.shape[1]*0.2)
		best_boxes[:,3] = best_boxes[:,3] + int(img.shape[0]*0.15)

	# detection result
	image = draw_scaled_boxes(img, best_boxes, best_probs, ["10", "1", "2", "3", "4", "5", "6", "7", "8", "9"])
	print("{}-boxes are detected.".format(len(best_boxes)))
	# plt.imshow(image)
	# plt.show()

	# append to result list
	result_dict = {"bbox":[], "label":[], "score":[]}
	try:
		result_dict["bbox"] = bbox_format(best_boxes).tolist()
		result_dict["label"] = np.argmax(np.array(best_probs), axis=1).tolist()
		result_dict["score"] = np.amax(np.array(best_probs), axis=1).tolist()
	except:
		pass
	result_list.append(result_dict)
	print(result_dict)
	print(len(result_list), '\n')
	
print(len(result_list), '\n')


# 4. result to file
import json
result_json = json.dumps(result_list)
with open('0856125.json', 'w') as f:
	json.dump(result_list, f)