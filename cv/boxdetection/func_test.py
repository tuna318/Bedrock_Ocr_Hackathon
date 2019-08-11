from box_extractor import box_extraction, fill_content
import cv2 

img = cv2.imread('test3.png')
content = [{'label': '1. Your name', 
            'fill_box': [468, 205, 411, 61], 
            'field': 'Tuna'}, 
            {'label': '2. Your permanent address', 
            'fill_box': [45, 442, 833, 61], 
            'field': 'Hanoi'}, 
            {'label': '3. Your email address', 
            'fill_box': [45, 640, 832, 61], 
            'field': 'tuna@example.com'}, 
            {'label': '4. Your mailing address (if different from above)', 
            'fill_box': [45, 748, 833, 60], 
            'field': 'teo@mail.com'}]

# fill_content(img, content)
box_extraction(img, 'Cropped/')

