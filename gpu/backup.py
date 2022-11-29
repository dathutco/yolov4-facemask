from flask import Flask, render_template, request
from flask import request
from datetime import datetime
# from time import gmtime, strftime
import os
import cv2
import numpy as np
from conMatrix import *
import xml.etree.ElementTree as ET
import firebase_admin
from firebase_admin import credentials, firestore
from firebase_admin import db
import time

# Create Flask Server Backend
app = Flask(__name__)

# load label
app.config['UPLOAD_FOLDER'] = "RecievedImg"
app.config['LABEL'] = "RecievedLabel"
app.config['VIDEO'] = "RecievedVideo"

cred = credentials.Certificate('./authentication.json')
default_app = firebase_admin.initialize_app(cred, {
    'databaseURL': "https://project-realtime-161a1-default-rtdb.firebaseio.com/"})

ref = db.reference("/recognizations/face_mark")

annotations = "./conMatrix/annotations"
dirMask = "conMatrix/mask"
dirNoMask = "conMatrix/nomask"

objectInFireBase = []


def makeDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


makeDir(app.config['UPLOAD_FOLDER'])
makeDir(app.config['LABEL'])
makeDir(app.config['VIDEO'])

formatDatetime = '%d-%m-%Y_%H-%M-%S-%f'
skipTime = 4
classes_file = "data/obj.names"
with open(classes_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
# color green vs red
colors = [(0, 255, 0), (0, 0, 255)]
# file model vs config
modelcfg = "cfg/yolov4-tiny-custom.cfg"
weight = "Model/best.weights"
# Load model
net = cv2.dnn.readNet(weight, modelcfg)
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


def saveFile(dir, file, name, extension):
    path_to_save = os.path.join(dir, f"{name}.{extension}")
    try:
        cv2.imwrite(path_to_save, file)
    except:
        file.save(path_to_save)
    return path_to_save


def detect(iH, iW, outs):
    class_ids = []
    confidences = []
    boxes = []
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
    return class_ids, confidences, boxes

# draw


def draw(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])+" (" + str(round(confidence*100, 2)) + "%)"
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), colors[class_id], 2)
    cv2.putText(img, label, (x-10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_id], 2)


def getLabel(dir, file):
    image = cv2.imread(dir+"/"+file)

    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    classids, _, _ = detect(image.shape[:2][0], image.shape[:2][1], outs)

    tree = ET.parse(f'{annotations}/{file[:-4]}.xml')
    object = tree.find('object')
    try:
        return str(classes[int(classids[0])]), object.find("name").text
    except:
        return str(classes[1]), object.find("name").text


# App default
@app.route('/', methods=['POST', 'GET'])
def image():
    if request.method == 'POST':
        # Take request
        name = f"{datetime.now().strftime(formatDatetime)}"
        print(f"from: {name}")
        img = request.files['file']
        file_bytes = np.fromfile(img, np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        res = []

        print(f"to:   {datetime.now().strftime(formatDatetime)}")
        # build
        blob = cv2.dnn.blobFromImage(
            image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        print(f"to:   {datetime.now().strftime(formatDatetime)}")
        # detect
        class_ids, confidences, boxes = detect(
            image.shape[:2][0], image.shape[:2][1], outs)
        # take index in list
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        info = ""
        for i in indexes:
            lst = []
            # append label
            label = str(classes[class_ids[i]])
            lst.append(label)
            # append x, y, weight, height
            x, y, w, h = boxes[i]
            lst.extend(boxes[i])
            # append confidences
            lst.append(confidences[i])
            res.append(lst)

            info += f"{class_ids[i]} {x} {y} {w} {h}\n"
            nowTime = int(time.time())
            yu = [x, y, w, h, label, nowTime, img.filename]

            objectInFireBase.append(x)
            objectInFireBase.append(y)
            objectInFireBase.append(w)
            objectInFireBase.append(h)
            objectInFireBase.append(label)
            objectInFireBase.append(nowTime)
            objectInFireBase.append(img.filename)
            # insertData(x, y, w, h, label, nowTime, img.filename)

        pathsave = os.path.join(app.config['LABEL'], f"{name}.txt")

        if os.listdir(app.config['UPLOAD_FOLDER']):
            last = datetime.strptime(os.listdir(
                app.config['UPLOAD_FOLDER'])[-1].split('.')[0], formatDatetime)
        else:
            last = datetime.min
        now = datetime.strptime(name, formatDatetime)

        if info != "" and [value for value in confidences if value < 0.9] == [] and (now-last).seconds > skipTime:
            # save image
            path_to_save = saveFile(
                app.config['UPLOAD_FOLDER'], image, name, "jpg")
            re = cv2.imread(path_to_save)
            f = open(pathsave, "w")
            # save label
            f.write(info)
            f.close()
        print(f"to:   {datetime.now().strftime(formatDatetime)}")
        return res
    return {}


@app.route('/resetValidate', methods=['GET'])
def resetValidate():
    data = {'predict': [], 'label': []}
    df = pd.DataFrame(data=data)

    mask = os.listdir(dirMask)
    nomask = os.listdir(dirNoMask)
    for i, j in zip(mask, nomask):
        rowi = getLabel(dirMask, i)
        rowj = getLabel(dirNoMask, j)
        df.loc[len(df.index)] = rowi
        df.loc[len(df.index)] = rowj
    """
    Then save to table
    """
    json_data = df.to_json(orient='values')
    return json_data


@app.route('/user-confirm-label', methods=['GET'])
def userConfirm():
    predict = request.args.get('predict')
    key = bool(request.args.get('key'))
    if (predict == 'without_mask'):
        if (key == True):
            objectInFireBase.append('without_mask')
        else:
            objectInFireBase.append('with_mask')
    else:
        if (key == True):
            objectInFireBase.append('with_mask')
        else:
            objectInFireBase.append('without_mask')
    insertData(objectInFireBase[0], objectInFireBase[1], objectInFireBase[2], objectInFireBase[3],
               objectInFireBase[4], objectInFireBase[5], objectInFireBase[6], objectInFireBase[7])
    return "Confirm successfully"


@app.route('/get-all-data', methods=['GET'])
def getAllData():
    objectData = ref.get()
    listData = objectData.values()
    mark = 0
    withoutMark = 0
    for data in listData:
        if data['label'] == 'without_mask':
            withoutMark = withoutMark + 1
        elif data['label'] == 'with_mask':
            mark = mark + 1

    return {
        "mark": mark,
        "withoutMark": withoutMark,

    }


@app.route('/get-data-by-time', methods=['GET'])
def getDataByTime():
    objectData = ref.get()
    listData = objectData.values()
    type = 'DAY'
    # listDataMask = []
    # listDataWithoutMask = []
    if type == 'DAY':
        # dateTimeEnd = datetime.combine(datetime.now(), time.max)
        # dateUnixTimeEnd = int(dateTimeEnd.timestamp())
        dateTimeStart = datetime.combine(datetime.now(), time.min)
        dateUnixTimeStart = int(dateTimeStart.timestamp())
        t6y = 0
        t12y = 0
        t18y = 0
        t24y = 0
        t6n = 0
        t12n = 0
        t18n = 0
        t24n = 0
        for data in listData:
            if data["time"] >= dateUnixTimeStart and data["time"] <= (dateUnixTimeStart + 3600 * 6):
                if data['label'] == 'without_mask':
                    t6n = t6n + 1
                elif data['label'] == 'with_mask':
                    t6y = t6y + 1
            elif data["time"] >= (dateUnixTimeStart + 3600 * 6) and data["time"] <= (dateUnixTimeStart + 3600 * 12):
                if data['label'] == 'without_mask':
                    t12n = t12n + 1
                elif data['label'] == 'with_mask':
                    t12y = t12y + 1
            elif data["time"] >= (dateUnixTimeStart + 3600 * 12) and data["time"] <= (dateUnixTimeStart + 3600 * 18):
                if data['label'] == 'without_mask':
                    t18n = t18n + 1
                elif data['label'] == 'with_mask':
                    t18y = t18y + 1
            elif data["time"] >= (dateUnixTimeStart + 3600 * 18) and data["time"] <= (dateUnixTimeStart + 3600 * 24):
                if data['label'] == 'without_mask':
                    t24n = t24n + 1
                elif data['label'] == 'with_mask':
                    t24y = t24y + 1
    listDataMask = [0, t6y, t12y, t18y, t24y]
    listDataWithoutMask = [0, t6n, t12n, t18n, t24n]
    return {
        "mask": str(listDataMask),
        "withoutMask": str(listDataWithoutMask)
    }


@app.route("/chart")
def home():
    return render_template("chart.html")


def insertData(x, y, w, h, label, nowTime, img, confirmedLable):
    objectInFireBase.clear()
    ref.push().set({
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'label': label,
        'time': nowTime,
        'image': img,
        'confirmedLable': confirmedLable
    })


@app.route('/score', methods=['GET'])
def score():
    data = {'predict': ["with_mask", "with_mask", "without_mask", "with_mask", "with_mask", "without_mask", "without_mask"], 'label': [
        "with_mask", "with_mask", "without_mask", "with_mask", "without_mask", "without_mask", "without_mask"]}
    df = pd.DataFrame(data=data)

    matrix = ConfusionMatrix(df)
    acc, recall, precision, f1 = matrix.allScore()
    return {"accuracy": acc, "recall": recall, "precision": precision, "f1-score": f1}


@app.route('/validate', methods=['GET'])
def validate():
    predict = request.form.get('predict')
    key = bool(request.form.get('key'))

    if (key == True):
        label = predict
    else:
        if (predict == str(classes[0])):
            label = str(classes[1])
        else:
            label = str(classes[0])

    """
    edit database
    """
    data = {'predict': ["with_mask"], 'label': ["with_mask"]}
    df = pd.DataFrame(data=data)
    df.loc[len(df.index)] = [predict, label]

    return [predict, label]


@app.route('/video', methods=['POST'])
def video():
    name = f"{datetime.now().strftime(formatDatetime)}"
    print(f"from: {name}")
    vid = request.files['file']

    path_to_save = saveFile(
        app.config['VIDEO'], vid, vid.filename.split('.')[0], "mp4")
    # path_to_save = saveFile(app.config['VIDEO'],vid, name, "mp4")
    video = cv2.VideoCapture(path_to_save)

    w = video.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = video.get(cv2.CAP_PROP_FPS)
    dir = app.config['VIDEO']

    out = cv2.VideoWriter(f'{dir}/{name}.mp4', -1, fps, (int(w), int(h)))

    while True:
        _, frame = video.read()
        try:
            blob = cv2.dnn.blobFromImage(
                frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            class_ids, confidences, boxes = detect(
                frame.shape[:2][0], frame.shape[:2][1], outs)
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        except:
            break
        # draw
        for i in indexes:
            x, y, w, h = boxes[i]
            # draw box
            draw(frame, class_ids[i], confidences[i], round(
                x), round(y), round(x + w), round(y + h))
        out.write(frame)
        cv2.imshow("Image", frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    video.release()
    out.release()
    cv2.destroyAllWindows()
    newPath = os.path.join(
        dir, f"{name}.{vid.filename.split('.')[-1]}").replace("\\", "/")
    print(f"to:   {datetime.now().strftime(formatDatetime)}")
    if os.path.exists(newPath):
        if os.path.exists(path_to_save):
            os.remove(path_to_save)
        return request.host_url+newPath
    return "Cancel"


# Start Backend
if __name__ == '__main__':
    app.run(port=30701, debug=True)
