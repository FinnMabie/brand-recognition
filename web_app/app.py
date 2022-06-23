from flask import Flask, Response, render_template
import numpy as np
import cv2
from helpers.helpers import *
import time
import webbrowser


app = Flask(__name__)
cap = cv2.VideoCapture(0)

labels, loaded_model = load_model()

@app.route('/')
def index():
    return render_template("index.html")

def gen(video):
    font = cv2.FONT_HERSHEY_PLAIN
    color=(255, 0, 0)

    i = 0
    while True:
        ret, frame = video.read()

        img = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        img = process_image(img)
        img = img.reshape(1,224,224,3)
        
        if i % 20 == 0:
            # predict and get three top most likely brands
            prediction = loaded_model.predict(img)
            top_values_index = sorted(range(len(prediction[0])), key=lambda i: prediction[0][i])[-3:]

            # grab labels
            top3ind = top_values_index[::-1]
            top3 = [labels[top3ind[0]], labels[top3ind[1]], labels[top3ind[2]]]
            p_val = np.max(prediction)
            prediction = labels[np.argmax(prediction)]
            if p_val >= 1.0:
                x, y, w, h = 20, 20, 250, 80
                cv2.putText(frame, prediction, (x, y + 30), font, 3, color, 3)
                
        ret, jpeg = cv2.imencode('.jpg', frame)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route("/video")
def video():
    global cap
    return Response(gen(cap), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2204, threaded=True)

