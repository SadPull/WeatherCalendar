import cv2,numpy,os,json
import numpy as np
labels,faces =[],[]
file = "haarcascades/lbpcascade_frontalface_improved.xml"
face_cascade = cv2.CascadeClassifier(file)
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

def detect_face(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.2,5,minSize=(20,20))
    if (len(faces)==0):
        return None
    (x,y,w,h)=faces[0]
    return gray[y:y+w,x:x+h]

def read_face(label,images_path):
    print("trainning:",label,images_path)
    files = os.listdir(images_path)
    for file in files:
        if file.startswith('.'):
            continue
        image = cv2.imdecode(np.fromfile(os.path.join('faceSource', file), dtype=np.uint8), cv2.IMREAD_COLOR)

        face = detect_face(image)
        # print(face)
        # with open("data.json", encoding="utf-8") as f:
        #     data = json.load(f)
        #
        # face = np.array(data['data'])
        # print(face)
        if face is not None:
            face = cv2.resize(face,(256,256))
            faces.append(face)
            labels.append(label)

if __name__=='__main__':
    # 存储用于训练文件的位置，根据实际情况修改
    read_face(1,"faceSource/")
    #read_face(2,"C:/Users/polis/Desktop/facial_recognition_for_use/training/iron_man/")
    face_recognizer.train(faces,numpy.array(labels))
    #训练数据存储文件的名称，可以修改
    face_recognizer.save('trainner_wmz.yml')