import config

from flask import Flask, Response, request, url_for, send_from_directory
from flask_restful import reqparse, abort, Api, Resource
import werkzeug
from flask import request
import cv2
import numpy as np
from functools import wraps
import pytesseract
import jsonpickle
from flask_cors import CORS, cross_origin
from cv.boxdetection.box_extractor import box_extraction, fill_content
from werkzeug.utils import secure_filename
import os 

app = Flask(__name__)
api = Api(app)
CORS(app)

app.config['UPLOADED_PHOTOS'] = 'static'
ALLOWED_EXTENSIONS = ["jpg", "png", "JPG"]

class OcrScan(Resource):
    def __init__(self):
        # self.img = None
        pass

    def check_support_file_type(self, filename):
        return '.' in filename and \
            filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
    def get(self):
        return "hello Hoang"

    def post(self):
        # return "Huhu"
        try:
            parser = reqparse.RequestParser()
            parser.add_argument('photo', type=werkzeug.FileStorage, location='files')
            args = parser.parse_args()
        except:
            abort(403, description='Server Error')

        if not self.check_support_file_type(args['photo'].filename):
            abort(401, description="File type error, file type doesn't support!")
        
        # npimg = np.fromstring(args['photo'].read(), np.uint8)
        img = request.files['photo']
        filename = secure_filename(img.filename)
        npimg = np.fromstring(img.stream.read(), np.uint8)
        
        filename = secure_filename(img.filename)
        img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
        # img.save(os.path.join(app.config['UPLOADED_PHOTOS'], filename))
        cv2.imwrite(os.path.join(app.config['UPLOADED_PHOTOS'], filename), img)
        

        response = box_extraction(img, 'cv/boxdetection/Cropped/')
        responses = {}
        responses["data"] = response
        responses["img_uri"] = url_for(app.config['UPLOADED_PHOTOS'],  filename =filename)
        print(responses["img_uri"])
        # text = pytesseract.image_to_string(image, lang='vie')

        # response = {'Text': text}
        response_pickled = jsonpickle.encode(responses)

        return Response(response=response_pickled, status=200, mimetype="application/json")

class ProcessImage(Resource):
    def get(self):
        pass
        

    def post(self):
        contents = request.json['data']
        print(request.json)
        img_uri = request.json['img_uri']
        filename = img_uri.split('/')[-1]
        image = cv2.imread(img_uri[1:])
        
        upload_path, download_path = fill_content(image, contents, filename)

        return {"upload": upload_path, "download": download_path}
        
    
    def delete(self):
        pass
    

@app.route('/uploads/<path:path>', methods=['GET'])
def upload_file(path):
    return send_from_directory('data',
                               path, as_attachment=False)

@app.route('/downloads/<path:filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory('data',
                               filename, as_attachment=True)

api.add_resource(OcrScan, '/api/ocr')
api.add_resource(ProcessImage, '/api/form')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)