from asyncio.windows_events import NULL
from PIL import Image
from urllib import response
from flask import Flask
# from flask_cors import CORS, cross_origin
from flask import request
from time import gmtime, strftime
import os
# import yolo
import cv2

# Khởi tạo Flask Server Backend
app = Flask(__name__)

# Apply Flask CORS
# CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = "static"
# yolov6_model = my_yolov6.my_yolov6("weights/fire_detect.pt", "cpu", "data/mydataset.yaml", 640, False)

modelcfg="cfg/yolov4.cfg"
weight="Model/yolov4-custom_best.weights"
# net=cv2.dnn.readNetFromDarknet(modelcfg,weight )


@app.route('/', methods=['POST'] )
def image():
    img = request.files['file']
    if img:
        print(type(img))
        path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
        print("Save = ", path_to_save)
        img.save(path_to_save)
        # frame=Image.open(img)

        frames = cv2.imread(img)
        # frame, no_object = yolov6_model.infer(frame)
        # del frame
        # if no_object >0:
        #     cv2.imwrite(path_to_save, frame)
        return "1"

    return "empty"

# Start Backend
if __name__ == '__main__':
    app.run()