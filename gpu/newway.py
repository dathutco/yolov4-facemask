from flask import Flask
from flask import request
from datetime import datetime
# from time import gmtime, strftime
import os
import cv2
import numpy as np
from conMatrix import *
import xml.etree.ElementTree as ET
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

# Create Flask Server Backend
app = Flask(__name__)

## load label
app.config['UPLOAD_FOLDER'] = "RecievedImg"
app.config['LABEL'] = "RecievedLabel"
app.config['VIDEO'] = "RecievedVideo"

# cred = credentials.Certificate('./credential.json')
# default_app = firebase_admin.initialize_app(cred, {
#     'databaseURL': "https://fir-3704f-default-rtdb.firebaseio.com/"})

# ref = db.reference("/Validation")


annotations="./conMatrix/annotations"
dirMask="conMatrix/mask"
dirNoMask="conMatrix/nomask"

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
modelcfg="cfg/yolov4-tiny-custom.cfg"
weight="Model/best.weights"

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


## draw
def draw(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])+" ("+ str(round(confidence*100,2)) +"%)"
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), colors[class_id], 2)
    cv2.putText(img, label, (x-10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id], 2)


def getLabel(dir,file):
    image=cv2.imread(dir+"/"+file)
    classids,_,_ = model.detect(image, 0.5, 0.4)
    tree = ET.parse(f'{annotations}/{file[:-4]}.xml')
    object=tree.find('object')
    try:
        return str(classes[int(classids[0])]),object.find("name").text
    except:
        return str(classes[1]),object.find("name").text

        
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


@app.route('/resetValidate', methods=['GET'] )
def resetValidate():
    data = {'predict': [], 'label': []}
    df = pd.DataFrame(data=data)

    mask=os.listdir(dirMask)    
    nomask=os.listdir(dirNoMask)
    for i,j in zip(mask,nomask):
        rowi=getLabel(dirMask,i)
        rowj=getLabel(dirNoMask,j)
        df.loc[len(df.index)] = rowi
        df.loc[len(df.index)] = rowj

    # for predict,label in df.iterrows():
    #     ref.push().set({
    #         'predict': predict,
    #         'label': label
    #     })
    json_data = df.to_json(orient='values')
    return json_data
    # return "sucess"


@app.route('/score', methods=['GET'] )
def score():
    data = {'predict': ["with_mask","with_mask","without_mask","with_mask","with_mask","without_mask","without_mask"], 'label': ["with_mask","with_mask","without_mask","with_mask","without_mask","without_mask","without_mask"]}
    df = pd.DataFrame(data=data)

    matrix=ConfusionMatrix(df)
    acc,recall,precision,f1=matrix.allScore()
    return {"accuracy":acc,"recall":recall,"precision":precision,"f1-score":f1}


@app.route('/validate', methods=['GET'] )
def validate():
    predict = request.form.get('predict')
    key = bool(request.form.get('key'))

    if(key==True):
        label=predict
    else:
        if(predict==str(classes[0])):
            label=str(classes[1])
        else:
            label=str(classes[0])
    
    """
    edit database
    """
    data = {'predict': ["with_mask"], 'label': ["with_mask"]}
    df = pd.DataFrame(data=data)
    df.loc[len(df.index)] = [predict,label]
    
    return [predict,label]


@app.route('/video', methods=['POST'] )
def video():
    name=f"{datetime.now().strftime(formatDatetime)}"
    print(f"from: {name}") 
    vid = request.files['file']

    path_to_save = saveFile(app.config['VIDEO'], vid, vid.filename.split('.')[0], "mp4")
    # path_to_save = saveFile(app.config['VIDEO'],vid, name, "mp4")

    video = cv2.VideoCapture(path_to_save)

    w = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = video.get(cv2.CAP_PROP_FPS) 
    dir=app.config['VIDEO']
    
    out = cv2.VideoWriter(f'{dir}/{name}.mp4', -1, fps, (int(w),int(h)))

    while True:
        _, frame = video.read()

        try:
            classids, scores, boxes = model.detect(frame, 0.5, 0.4)
        except:
            break
        ## take index in list 
        info=""
        for (classid, score, box) in zip(classids, scores, boxes):

            # x, y, weight, height
            x, y, w, h=[float(f) for f in box]
            ### draw box
            draw(frame, int(classid), float(score), round(x), round(y), round(x + w), round(y + h))
        
        out.write(frame)
        cv2.imshow("Image", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    video.release()
    out.release()
    cv2.destroyAllWindows()
    newPath=os.path.join(dir, f"{name}.{vid.filename.split('.')[-1]}").replace("\\","/")
    print(f"to:   {datetime.now().strftime(formatDatetime)}")
    if os.path.exists(newPath):
        if os.path.exists(path_to_save):
            os.remove(path_to_save)
        return request.host_url+newPath
    return "Cancel"
    

# Start Backend
if __name__ == '__main__':
    app.run(port=30701,debug=True)

