import requests
import json
import cv2

addr = 'http://localhost:5000'
test_url = addr + '/api/ocr/scan'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

# img = cv2.imread('data/vn.png')
# encode image as jpeg
# _, img_encoded = cv2.imencode('.jpg', img)
# sent http request with image and receive response
files = {'file': open('data/vn.png', 'rb')}

response = requests.post(test_url, files=files)
# decode response
print(response.text)

