from flask import Flask,request,jsonify
import face_recognition
import numpy as np
import cv2
app =Flask(__name__)

@app.route('/')

def home():
    return "Hello world"

@app.route('/face_match',methods=['POST'])
def face_match():
    file = request.files['image1'].read()
    npimg = np.fromstring(file, np.uint8)
    img1 = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    rgb_img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
    img1_encoding=face_recognition.face_encodings(rgb_img1)[0]

    file2 = request.files['image2'].read()  ## byte file
    npimg2 = np.fromstring(file2, np.uint8)
    img2 = cv2.imdecode(npimg2, cv2.IMREAD_COLOR)
    rgb_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    img2_encoding = face_recognition.face_encodings(rgb_img2)[0]

    result=face_recognition.compare_faces([img1_encoding],img2_encoding)
    res={'face_matched':str(result[0])}
    return jsonify(res)


if __name__=='__main__':
    app.run(debug=True)
