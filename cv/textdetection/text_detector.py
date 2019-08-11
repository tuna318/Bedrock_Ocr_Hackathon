import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/home/tuna/AI/Computer_Vision/OCR/cv/textdetection/gg_api/TextRecognition-6510c46167e2.json"


import cv2
import numpy as np
import math


def detect_text(img):
	"""Detects text in the file."""
	# img = cv2.imread(path, 0)
	from google.cloud import vision
	client = vision.ImageAnnotatorClient()

	# with open(path, 'rb') as image_file:
	# image_file = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
	image_file = cv2.imencode('.jpg', img)[1].tostring()
	content = image_file

	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	image = vision.types.Image(content=content)

	response = client.text_detection(image=image)
	texts = response.text_annotations
	print('Texts:')

	full_text = ""

	text_boxes = []
	big_box = []
	sentence = ""
	sentence_dict = {}
	for text in texts[1:]:
		# print('\n"{}"'.format(text.description))

		vertices = (['({},{})'.format(vertex.x, vertex.y)
					for vertex in text.bounding_poly.vertices])
					
		current_word_box = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
		# print(text_box)
		# text_boxes.append((text_box[0][0], text_box[0][1], text_box[1][0] - text_box[0][0], text_box[2][1] - text_box[0][1]))
		# startX, endX, startY, endY = text_box[0][0], text_box[1][0], text_box[0][1], text_box[2][1]
		# print("current: ", current_word_box)
		

		if len(big_box) == 0:
			big_box = current_word_box
			sentence = text.description
		else:
			prev_word_box = big_box
			space_threshold = math.sqrt((prev_word_box[0][1] - prev_word_box[3][1])**2 + (prev_word_box[0][0] - prev_word_box[3][0])**2)*1.2
			prev_word_len =  math.sqrt((prev_word_box[3][1] - prev_word_box[2][1])**2 + (prev_word_box[3][0] - prev_word_box[2][0])**2)
			word_space = math.sqrt((current_word_box[3][1] - prev_word_box[2][1])**2 + (current_word_box[3][0] - prev_word_box[2][0])**2)
			word_start_to_start_dis = math.sqrt((current_word_box[3][1] - prev_word_box[3][1])**2 + (current_word_box[3][0] - prev_word_box[3][0])**2)

			# print("--------------------------")
			# print(current_word_box[2][1], prev_word_box[3][1], current_word_box[2][0], prev_word_box[3][0])
			# print("word_space: ", word_space, "space_threshold: ", space_threshold)
			# print("prev text: ", sentence, "current_word:",  text.description)
			# print(current_word_box)
			# print("---------------------------")
			# sentence = text.description
			# big_box = current_word_box

			min_startX = min(prev_word_box[0][1], current_word_box[0][1], prev_word_box[1][1], current_word_box[1][1] )
			max_endX = max(prev_word_box[2][1], current_word_box[2][1], prev_word_box[3][1], current_word_box[3][1])
			min_startY = min(prev_word_box[0][0], current_word_box[0][0], current_word_box[1][0], prev_word_box[1][0])
			max_endY = max(prev_word_box[2][0], current_word_box[2][0], prev_word_box[3][0], current_word_box[3][0])

			if (space_threshold > word_space) and (prev_word_len + word_space - word_start_to_start_dis < word_space):
				big_box = [(min_startY, min_startX), (max_endY, min_startX), (max_endY, max_endX), (min_startY, max_endX)]
				sentence = sentence + " " + text.description
				
			else:
				box = (big_box[0][0], big_box[0][1], big_box[2][0] - big_box[0][0], big_box[2][1] - big_box[0][1])
				text_boxes.append({"box": box, "sentence": sentence})
				sentence_dict[sentence] = box

				cv2.rectangle(img, (big_box[0][0], big_box[0][1]), (big_box[2][0], big_box[2][1]), (255, 255, 255), -1)
				print("big-text: ", sentence)
				sentence = text.description				
				big_box = current_word_box
	box = (big_box[0][0], big_box[0][1], big_box[2][0] - big_box[0][0], big_box[2][1] - big_box[0][1])
	text_boxes.append({"box": box, "sentence": sentence})			
	sentence_dict[sentence] = box
	cv2.rectangle(img, (big_box[0][0], big_box[0][1]), (big_box[2][0], big_box[2][1]), (255, 255, 255), -1)
	print("big-text: ", sentence)

		# cv2.rectangle(img, (startX, startY), (endX, endY), (255, 255, 255), -1)

		# print('bounds: {}'.format(','.join(vertices)))
		# print(current_word_box)
	
	# print(img.shape)
	# cv2.imshow("Text Detection", img)
	# cv2.waitKey(0)
	return img, text_boxes, sentence_dict

# detect_text('test.png')