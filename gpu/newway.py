# from PIL import Image
from asyncio.windows_events import NULL
import socket
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
app.config['UPLOAD_FOLDER'] = "RecievedImg"
app.config['LABEL'] = "RecievedLabel"
app.config['VIDEO'] = "RecievedVideo"

def makeDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

makeDir(app.config['UPLOAD_FOLDER'])
makeDir(app.config['LABEL'])
makeDir(app.config['VIDEO'])

formatDatetime='%d-%m-%Y_%H-%M-%S-%f'
skipTime=4
classes_file = "data/obj.names"
with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

### color green vs red
colors=[(0, 255, 0),(0, 0, 255)]
##file model vs config
modelcfg="cfg/yolov4.cfg"
weight="Model/yolov4-custom_best.weights"

## Load model
net=cv2.dnn.readNet(weight,modelcfg)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def saveFile(dir,file,name,extension):
    path_to_save = os.path.join(dir, f"{name}.{extension}")
    try:
        cv2.imwrite(path_to_save,file)
    except:
        file.save(path_to_save)
    return path_to_save

def detect(iH,iW,outs):
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.7:
                center_x = float(detection[0] * iW)
                center_y = float(detection[1] * iH)
                w = float(detection[2] * iW)
                h = float(detection[3] * iH)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    return class_ids,confidences,boxes

## draw
def draw(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])+" ("+ str(round(confidence*100,2)) +"%)"
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), colors[class_id], 2)
    cv2.putText(img, label, (x-10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id], 2)

### App default
@app.route('/', methods=['POST','GET'] )
def image():
    if request.method=='POST':
        ##Take request
        name=f"{datetime.now().strftime(formatDatetime)}"
        print(f"from: {name}")
        img = request.files['file']

        file_bytes = np.fromfile(img, np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        res=[]
        
        print(f"to:   {datetime.now().strftime(formatDatetime)}")
        ## detect
        classids, scores, boxes = model.detect(image, 0.5, 0.4)
        ## take index in list 
        info=""
        for (classid, score, box) in zip(classids, scores, boxes):
            lst=[]
            ## append label
            lst.append(str(classes[int(classid)]))
            # append x, y, weight, height
            x, y, w, h=[float(f) for f in box]
            lst.extend([x,y,w,h])
            ## append confidences
            lst.append(float(score))
            res.append(lst)

            info+=f"{int(classid)} {x} {y} {w} {h}\n"

        pathsave = os.path.join(app.config['LABEL'], f"{name}.txt")

        if os.listdir(app.config['UPLOAD_FOLDER']):
            last=datetime.strptime(os.listdir(app.config['UPLOAD_FOLDER'])[-1].split('.')[0], formatDatetime)
        else:
            last=datetime.min
        now=datetime.strptime(name, formatDatetime)
        
        ### Consider label!=NULL,confidences>=0.9 and accept time to write new image
        if info!="" and [value for value in scores if value<0.9]==[] and (now-last).seconds>skipTime:
            print(f"collected image with name {name}.jpg and label with name {name}.txt")
            ##save image
            path_to_save = saveFile(app.config['UPLOAD_FOLDER'],image,name, "jpg")
            f = open(pathsave, "w")
            # save label
            f.write(info)
            f.close()
        print(f"to:   {datetime.now().strftime(formatDatetime)}")
        return res
    return {}

@app.route('/video', methods=['POST'] )
def video():
    name=f"{datetime.now().strftime(formatDatetime)}"
    print(f"from: {name}") 
    vid = request.files['file']

    path_to_save = saveFile(app.config['VIDEO'], vid, vid.filename.split('.')[0], "mp4")
    # path_to_save = saveFile(app.config['VIDEO'],vid, name, "mp4")

    video = cv2.VideoCapture(path_to_save)

    while True:
        _, frame = video.read()

        classids, scores, boxes = model.detect(frame, 0.5, 0.4)
        ## take index in list 
        info=""
        for (classid, score, box) in zip(classids, scores, boxes):

            # x, y, weight, height
            x, y, w, h=[float(f) for f in box]
            ### draw box
            draw(frame, int(classid), float(score), round(x), round(y), round(x + w), round(y + h))
        
        cv2.imshow("Image", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    video.release()
    newPath=os.path.join(app.config['VIDEO'], f"{name}.{vid.filename.split('.')[-1]}").replace("\\","/")

    # cv2.imwrite(newPath, video)

    print(f"to:   {datetime.now().strftime(formatDatetime)}")
    if not os.path.exists(newPath):
        return request.host_url+newPath
    return "Cancel"

# Start Backend
if __name__ == '__main__':
    app.run(port=30701)