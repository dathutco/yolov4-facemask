import json
from flask import Flask, flash, request, redirect, url_for, render_template, Response
import cv2
import requests
from werkzeug.utils import secure_filename
import os
app = Flask(__name__)

address = "http://127.0.0.1:30701"
colors = {"with_mask": (0, 255, 0), "without_mask": (0, 0, 255)}
WITHOUT_MASK = "Không đeo khẩu trang"
WITH_MASK = "Đeo khẩu trang"

UPLOAD_FOLDER = 'C:/Users/tranq/Downloads/yolov4-facemask/gpu/static/img'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def draw(img, label, confidence, x, y, x_plus_w, y_plus_h):
    txt = f"{label} ({str(round(confidence*100,2))}%)"
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), colors.get(label), 2)
    cv2.putText(img, txt, (x-10, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, colors.get(label), 2)


def process(rp, image):
    a = rp.split(']')
    for lst in a:
        if lst[2:] != "":
            lst = lst[2:].replace('\"', "").replace('[', "").split(",")
            label, x, y, w, h, confidence = lst
            label = label.strip()
            try:
                x = float(x)
                y = float(y)
                h = float(h)
                w = float(w)
                confidence = float(confidence)
            except:
                pass
            draw(image, label, confidence, round(x),
                 round(y), round(x + w), round(y + h))
    return image


def gen():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        _, im_with_type = cv2.imencode(".jpg", frame)
        byte_im = im_with_type.tobytes()
        files = {'file': byte_im}
        rp = requests.post(address, files=files)
        frame = process(rp.text, frame)

        if not ret:
            print("Error: failed to capture image")
            break

        cv2.imwrite('demo.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('demo.jpg', 'rb').read() + b'\r\n')


def generate_frames():
    camera = cv2.VideoCapture("some_m3u8_link")
    while True:
        # read the camera frame
        success, frame = camera.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('./index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/send', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # img_filename = upload_file(file)

    URL = "http://127.0.0.1:30701"
    params = request.files
    response = requests.post(URL, files=params)

    if file and allowed_file(file.filename):
        if response.status_code != 500 and response.ok:
            my_json = response.content.decode('utf8').replace("'", '"')
            data = json.loads(my_json)
            s = json.dumps(data, indent=4, sort_keys=True)
            listResponse = json.loads(s)
            lable = listResponse[0][0]
            img_filename = listResponse[0][6] + '.jpg'
            result = ""
            if lable == "without_mask":
                result = WITHOUT_MASK
            if lable == "with_mask":
                result = WITH_MASK

            return render_template('./index.html', filename=img_filename, result=result, lable=lable)
        return "<h2>Can not regnization</h2>"


@app.route('/user-confirm-label', methods=['GET'])
def userConfirm():
    predict = request.args.get('predict')
    key = bool(request.args.get('key'))
    return "hello"


def upload_file(file):
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return filename


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='img/' + filename), code=301)


@app.route('/video')
def webcamera():
    return render_template('./video.html')


if __name__ == '__main__':
    app.run(debug=True)
