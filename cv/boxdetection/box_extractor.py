import cv2
import numpy as np
import sys
sys.path.append('/home/tuna/AI/Computer_Vision/OCR/cv/textdetection/')
import os
import re
from ast import literal_eval

from text_detector import detect_text

font_scale = 2
font = cv2.FONT_HERSHEY_PLAIN

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def box_extraction(cv2_img, cropped_dir_path):

    # raw_img = cv2.imread(img_for_box_extraction_path)  # Read the image
    # img_path = os.path.abspath(img_for_box_extraction_path)
    img, text_boxes, sentence_dict = detect_text(cv2_img)
    img[img < 250] = 0
    (thresh, img_bin) = cv2.threshold(img, 128, 255,
                                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Thresholding the image
    img_bin = 255-img_bin  # Invert the image

    cv2.imwrite("Image_bin.jpg",img_bin)
   
    # Defining a kernel length
    kernel_length = np.array(img).shape[1]//70

    # Sentence segmentation
    sentences_segement = {}
    for text_box in text_boxes:
        try:
            seg_num = int(re.findall('\d+', text_box['sentence'])[0])
            sentences_segement[seg_num] = text_box['sentence']
        except:
            continue

    for id, sent in sentences_segement.items():
        print(id, sent)

     
    # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
    verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))
    # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
    hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
    # A kernel of (3 X 3) ones.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Morphological operation to detect verticle lines from an image
    img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
    verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)
    cv2.imwrite("verticle_lines.jpg",verticle_lines_img)

    # Morphological operation to detect horizontal lines from an image
    img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
    horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)
    cv2.imwrite("horizontal_lines.jpg",horizontal_lines_img)

    # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
    alpha = 0.5
    beta = 1.0 - alpha
    # This function helps to add two image with specific weight parameter to get a third image as summation of two image.
    img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
    img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
    (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # For Debugging
    # Enable this line to see verticle and horizontal lines in the image which is used to find boxes
    cv2.imwrite("img_final_bin.jpg",img_final_bin)
    # Find contours for image, which will detect all the boxes
    contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # Sort all the contours by top to bottom.
    (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")

    # <+================= Hardest thing ===============> #
    idx = 0
    # form_labels 
    form_labels = []
    for c in contours:
        # Returns the location and width,height for every contour
        x, y, w, h = cv2.boundingRect(c)
        
        senseg_last_id = len(sentences_segement)
        # If the box height is greater then 20, widht is >80, then only save it as a box in "cropped/" folder.
        if (w > 80 and h > 20) and w > 3*h:
            print("----------")
            no_text_between_segment_flag = False
            
            for i in range(1, len(sentence_dict)):
                try:
                    # up_seg_pos = sentence_dict[sentences_segement[i]][1]
                    # down_seg_pos = sentence_dict[sentences_segement[i+1]][1]
                    # print(up_seg_pos, y, down_seg_pos)
                    
                    for j in range(len(text_boxes)):
                        if text_boxes[j]['sentence'] == sentences_segement[i] and text_boxes[j+1]['sentence'] == sentences_segement[i+1]:
                            no_text_between_segment_flag = True
                    
                        no_text_between_segment_flag = True

                    if no_text_between_segment_flag == False:
                        continue
                    else:
                        up_seg_pos = sentence_dict[sentences_segement[i]][1]
                        down_seg_pos = sentence_dict[sentences_segement[i+1]][1]
                        if up_seg_pos < y and down_seg_pos > y:
                            # print(up_seg_pos, y, down_seg_pos)
                            
                            form_labels.append({"label": sentences_segement[i], "fill_box": [x, y, w, h], "field": ""})
                except:
                    continue
            if (len(text_boxes) != 0):
                if text_boxes[-1]['sentence'] == sentences_segement[senseg_last_id]:
                    if sentence_dict[text_boxes[-1]['sentence']][1] < y:
                        # print(sentence_dict[text_boxes[-1]['sentence']][1], y)
                        form_labels.append({"label": text_boxes[-1]['sentence'], "fill_box": [x, y, w, h], "field": ""})
                        
            new_img = img[y:y+h, x:x+w]
            text_offset_x = x
            text_offset_y = y
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 30), 2)

            for form_label in form_labels:
                pos = form_label['fill_box']
                label = form_label['label']
                text_offset_x = pos[0] + int(pos[3]/4)
                text_offset_y = pos[1] + int(pos[3]/1.7)
                cv2.putText(img, label, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=3)
            cv2.imwrite(cropped_dir_path+str(idx) + '.png', new_img)
                    

    # for text_box in text_boxes:
    #     cv2.rectangle(img, (text_box[0], text_box[1]), (text_box[0] + text_box[2], text_box[1] + text_box[3]), (0,255,0), -1)
    # For Debugging
    # Enable this line to see all contours.
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)
    cv2.imwrite("./Temp/img_contour.jpg", img)
    labeled = []
    filtered_form_labeled = []
    for form_label in form_labels:
        if form_label["label"] not in labeled:
            filtered_form_labeled.append(form_label)
            labeled.append(form_label["label"])
    print(filtered_form_labeled)
    return filtered_form_labeled


def fill_content(image, contents, filename):
    """
    content has the form:
    [{
        "label": label,
        "fill_box": [x,y,w,h],
        "field" : field
    }
    ..
    ]
    """

    for content in contents:
        box = content["fill_box"]
        field = content["field"]
        text_offset_x = box[0]+int(box[3]/4)
        text_offset_y = box[1] + int(box[3]/1.7)
        cv2.putText(image, field, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(255, 0, 0), thickness=3)
    # cv2.imshow("show", image)
    # absolute_path = os.path.abspath(img_path)
    data_path = 'data/' + filename
    cv2.imwrite(data_path, image)
    # cv2.waitKey(0)

    return '/uploads/' + filename, '/downloads/' + filename
        
    

    

# box_extraction("test.png", "Cropped/")