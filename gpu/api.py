# from PIL import Image
from urllib import response
from flask import Flask
from flask import request
from datetime import datetime
# from time import gmtime, strftime
import os
import cv2
import numpy as np

# Create Flask Server Backend
app = Flask(__name__)

## load label
app.config['UPLOAD_FOLDER'] = "static"
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

classes_file = "data/obj.names"
with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
##file model vs config
modelcfg="cfg/yolov4.cfg"
weight="Model/yolov4-custom_best.weights"
## Load model
net=cv2.dnn.readNetFromDarknet(modelcfg,weight)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

### App default
@app.route('/', methods=['POST'] )
def image():
    ##Take request
    print(f"from: {datetime.now().strftime('%d/%m/%Y %H:%M:%S.%f')}")
    
    img = request.files['file']
    class_ids = []
    confidences = []
    boxes = []
    res=[]
    path_to_save = os.path.join(app.config['UPLOAD_FOLDER'], img.filename)
    img.save(path_to_save)
    # image=Image.open(img)

    image = cv2.imread(path_to_save)
    (iH, iW) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image,1 / 255.0,(416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.7:
                center_x = int(detection[0] * iW)
                center_y = int(detection[1] * iH)
                w = int(detection[2] * iW)
                h = int(detection[3] * iH)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    for i in indices:
        lst=[]
        box = boxes[i]
        lst.append(str(classes[class_ids[i]]))
        #x
        lst.append(box[0])
        #y
        lst.append(box[1])
        #weight
        lst.append(box[2])
        #heigh
        lst.append(box[3])
        lst.append(confidences[i])
        res.append(lst)

    del class_ids
    del confidences
    del boxes
    print(f"to:   {datetime.now().strftime('%d/%m/%Y %H:%M:%S.%f')}")
    return res

# Start Backend
if __name__ == '__main__':
    app.run()