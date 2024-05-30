import cv2
#人脸特征数据
names = ('ww','WMZ')#可以继续添加别人的人脸
file = "haarcascades/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(file)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('weight.yml')
#打开摄像头
vc = cv2.VideoCapture(0)
#设置视频画面宽为480像素，高为320像素
vc.set(cv2.CAP_PROP_FRAME_WIDTH,480)
vc.set(cv2.CAP_PROP_FRAME_HEIGHT,320)
while True:
    #读取视频帧图像
    retval,frame = vc.read()
    if not retval or cv2.waitKey(16) & 0xFF == ord('q'):
        break
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)#BGR
        face = gray[y:y+w,x:x+h]
        face = cv2.resize(face,(256,256))

        label,confidence = recognizer.predict(face)
        confidence = 100-confidence

        if label>0 and confidence>30:
            print(label)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            text = '%s:%d'%(names[label],confidence)
            print(text)
            font = cv2.FONT_HERSHEY_PLAIN
            cv2.putText(frame,text,(x,y),font,2.5,(0,255,0),2)
        else:
            print("ok")
    cv2.imshow('Video',frame)
#关闭摄像头
vc.release()
cv2.destroyAllWindows()