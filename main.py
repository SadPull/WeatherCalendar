import shutil
import sys,os,datetime,requests,sqlite3,cv2,json,numpy as np
# 导入图形组件库
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
#导入做好的界面库
from 主界面 import Ui_MainWindow
from 弹窗 import Ui_MainWindow1
from 登录 import Ui_MainWindow2

import resources

#登录
class loginWindow(QMainWindow,Ui_MainWindow2):
    d = pyqtSignal(str)
    def __init__(self):
        #继承(QMainWindow,Ui_MainWindow)父类的属性
        super(loginWindow,self).__init__()
        #初始化界面组件
        self.setupUi(self)
        self.label.setScaledContents(True)
        self.cap = cv2.VideoCapture(0)

        # 人脸特征数据
        file = "haarcascades/haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(file)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read('weight.yml')
        self._allNames = {}
        # 存储用于训练文件的位置，根据实际情况修改
        for file in os.listdir('faceSource'):
            self._allNames[int(file.split(' ')[0])] = file.split(' ')[1]


        #设置定时器
        self.index = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.refrsh)
        self.timer.start()

    def refrsh(self):
        retval, frame  = self.cap.read()
        if retval:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.index += 1
            if self.index > 100:
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)  # BGR
                    face = gray[y:y + w, x:x + h]
                    face = cv2.resize(face, (256, 256))

                    label, confidence = self.recognizer.predict(face)
                    confidence = 100 - confidence

                    if label > 0 and confidence > 30:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                        text = '%s:%d' % (self._allNames[label], confidence)
                        font = cv2.FONT_HERSHEY_PLAIN
                        cv2.putText(frame, text, (x, y), font, 2.5, (0, 255, 0), 2)
                        self.d.emit(self._allNames[label])
                        print(text)
                    else:
                        print("ok")


            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.Qframe = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[1] * 3, QImage.Format_RGB888).scaled(401,241)
            self.label.setPixmap(QPixmap.fromImage(self.Qframe))
    def closeEvent(self, *args, **kwargs):
        self.cap.release()
        self.timer.stop()

#注册
class registerWindow(QMainWindow,Ui_MainWindow1):
    d = pyqtSignal(str)
    def __init__(self,conn,cursor):
        #继承(QMainWindow,Ui_MainWindow)父类的属性
        super(registerWindow,self).__init__()
        #初始化界面组件
        self.setupUi(self)
        if os.path.exists("temp.jpg"):
            os.remove('temp.jpg')
        self.conn, self.cursor = conn,cursor
        self.label.setScaledContents(True)

        self.cap = cv2.VideoCapture(0)

        # 人脸特征数据
        file = "haarcascades/haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(file)
        self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
        # 设置定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.refrsh)

        #录入
        self.pushButton_3.clicked.connect(self.loadIn)


    def loadIn(self):
        _name = self.lineEdit.text()
        if _name:
            self._dirPath = os.getcwd() + f'/faceSource/{len(os.listdir("faceSource")) + 1} {_name}'
            os.makedirs(self._dirPath)
            self.sampleNum = 0
            self.timer.start()

        else:
            QMessageBox.warning(self,"错误","存在未输入信息",QMessageBox.Yes)
    def refrsh(self):
        retval, frame  = self.cap.read()
        if retval:
            # grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faceRects = self.face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=3, minSize=(200, 200))

            for faceRect in faceRects:
                x, y, w, h = faceRect
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                f = cv2.resize(frame[y:y + h, x:x + w], (100, 100))
                # 后面编号可以改，用于使图像命名不重复
                # cv2.imwrite(self._dirPath+"/" + str(self.sampleNum) + ".jpg", f)
                # print()

                cv2.imencode('.jpg', f)[1].tofile(self._dirPath+"/" + str(self.sampleNum) + ".jpg")
                self.sampleNum = self.sampleNum + 1

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.Qframe = QImage(frame.data, frame.shape[1], frame.shape[0], frame.shape[1] * 3, QImage.Format_RGB888).scaled(401,241)
            self.label.setPixmap(QPixmap.fromImage(self.Qframe))

            if self.sampleNum > 20:
                labels, faces = [], []
                def detect_face(image):
                    image = cv2.imdecode(np.fromfile(image, dtype=np.uint8), flags=cv2.IMREAD_COLOR)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
                    if (len(faces) == 0):
                        return None
                    (x, y, w, h) = faces[0]
                    return gray[y:y + w, x:x + h]

                _allNames = {}
                # 存储用于训练文件的位置，根据实际情况修改
                for file in os.listdir('faceSource'):
                    for path in os.listdir(os.path.join('faceSource',file)):
                        image = os.path.join(os.path.join('faceSource',file),path)
                        face = detect_face(image)
                        if face is not None:
                            face = cv2.resize(face, (256, 256))
                            faces.append(face)
                            labels.append(int(file.split(' ')[0]))
                            _allNames[int(file.split(' ')[0])] = file.split(' ')[1]

                # read_face(2,"C:/Users/polis/Desktop/facial_recognition_for_use/training/iron_man/")
                self.face_recognizer.train(faces, np.array(labels))
                # 训练数据存储文件的名称，可以修改
                self.face_recognizer.save('weight.yml')

                _name = self.lineEdit.text()
                # 插入数据
                insert_sql = '''INSERT INTO dataSet (id,Name,times, things) VALUES (?,?,?,?);'''
                self.cursor.execute(insert_sql, ("1", _name, "", ""))
                self.cursor.execute(insert_sql, ("2", _name, "", ""))
                self.cursor.execute(insert_sql, ("3", _name, "", ""))
                self.cursor.execute(insert_sql, ("4", _name, "", ""))
                self.conn.commit()

                self.d.emit("ok")

                self.timer.stop()
                self.cap.release()


    def closeEvent(self, *args, **kwargs):
        self.cap.release()
        self.timer.stop()



class MainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self):
        #继承(QMainWindow,Ui_MainWindow)父类的属性
        super(MainWindow,self).__init__()
        #初始化界面组件
        self.setupUi(self)
        '''天气加载'''
        _key = 'SntlxmKl_y8S3D5Fe'
        _location = 'shanghai'
        url = f'''https://api.seniverse.com/v3/weather/daily.json?key={_key}&location={_location}&language=zh-Hans&unit=c&start=-1&days=5'''
        r = requests.get(url)
        # print(r.json())

        current_datetime = datetime.datetime.now()
        day_of_week = int(current_datetime.weekday())
        _curr = int(str(current_datetime).split("-")[2].split(" ")[0])

        self.label.setText(
            f'{str(current_datetime).split("-")[0]}.{str(current_datetime).split("-")[1]}.{str(current_datetime).split("-")[2].split(" ")[0]}')
        _weeksToNum = {
            0: "星期一",
            1: "星期二",
            2: "星期三",
            3: "星期四",
            4: "星期五",
            5: "星期六",
            6: "星期日",
        }
        self.label.setText(_weeksToNum[day_of_week])
        _day = r.json()['results'][0]['daily'][0]['date']
        self.label_2.setText(f'{_day}')
        #当天
        #时间

        #天气
        # 当天
        _nowWeather = r.json()['results'][0]['daily'][0]['text_day']
        if _nowWeather == '晴':
            self.frame_7.setStyleSheet("image: url(:/button/img/buttom/晴天_sunny.svg);")
        elif _nowWeather == '小雨':
            self.frame_7.setStyleSheet("image: url(:/button/img/buttom/小雨_light-rain.svg);")
        elif _nowWeather == '大雨':
            self.frame_7.setStyleSheet("image: url(:/button/img/buttom/大雨_heavy-rain.svg);")
        elif _nowWeather == '多云':
            self.frame_7.setStyleSheet("image: url(:/button/img/buttom/多云_cloudy.svg);")
        elif _nowWeather == '小雨':
            self.frame_7.setStyleSheet("image: url(:/button/img/buttom/小雨_light-rain.svg);")
        elif _nowWeather == '小雪':
            self.frame_7.setStyleSheet("image: url(:/button/img/buttom/雪花_snowflake.svg);")
        elif _nowWeather == '大雪':
            self.frame_7.setStyleSheet("image: url(:/button/img/buttom/雪花_snowflake.svg);")
        elif _nowWeather == '雷雨':
            self.frame_7.setStyleSheet("image: url(:/button/img/buttom/雷雨_thunderstorm-one.svg);")
        elif _nowWeather == '大雾':
            self.frame_7.setStyleSheet("image: url(:/button/img/buttom/大雾_fog.svg);")
        # 温度范围
        _lowerTemp = r.json()['results'][0]['daily'][0]['low']
        _higherTemp = r.json()['results'][0]['daily'][0]['high']

        self.label_14.setText(f'{_nowWeather}')
        #温度范围
        self.label_3.setText(f'{_lowerTemp}°C- {_higherTemp}°C')


        self.label_12.setText(
            f'{_curr + 1}号')

        # 当天
        _nowWeather = r.json()['results'][0]['daily'][1]['text_day']
        if _nowWeather == '晴':
            self.frame_2.setStyleSheet("image: url(:/button/img/buttom晴天_sunny.svg);")
        elif _nowWeather == '小雨':
            self.frame_2.setStyleSheet("image: url(:/button/img/buttom/小雨_light-rain.svg);")
        elif _nowWeather == '大雨':
            self.frame_2.setStyleSheet("image: url(:/button/img/buttom/大雨_heavy-rain.svg);")
        elif _nowWeather == '多云':
            self.frame_2.setStyleSheet("image: url(:/button/img/buttom/多云_cloudy.svg);")
        elif _nowWeather == '小雨':
            self.frame_2.setStyleSheet("image: url(:/button/img/buttom/小雨_light-rain.svg);")
        elif _nowWeather == '小雪':
            self.frame_2.setStyleSheet("image: url(:/button/img/buttom/雪花_snowflake.svg);")
        elif _nowWeather == '大雪':
            self.frame_2.setStyleSheet("image: url(:/button/img/buttom/雪花_snowflake.svg);")
        elif _nowWeather == '雷雨':
            self.frame_2.setStyleSheet("image: url(:/button/img/buttom/雷雨_thunderstorm-one.svg);")
        elif _nowWeather == '大雾':
            self.frame_2.setStyleSheet("image: url(:/button/img/buttom/大雾_fog.svg);")

        # 温度范围
        _lowerTemp = r.json()['results'][0]['daily'][1]['low']
        _higherTemp = r.json()['results'][0]['daily'][1]['high']

        self.label_5.setText(f'{_nowWeather}')
        #温度范围
        self.label_4.setText(f'{_lowerTemp}°C- {_higherTemp}°C')


        self.label_13.setText(
            f'{_curr + 2}号')

        # 当天
        _nowWeather = r.json()['results'][0]['daily'][2]['text_day']
        if _nowWeather == '晴':
            self.frame_3.setStyleSheet("image: url(:/button/img/buttom/晴天_sunny.svg);")
        elif _nowWeather == '小雨':
            self.frame_3.setStyleSheet("image: url(:/button/img/buttom/小雨_light-rain.svg);")
        elif _nowWeather == '大雨':
            self.frame_3.setStyleSheet("image: url(:/button/img/buttom/大雨_heavy-rain.svg);")
        elif _nowWeather == '多云':
            self.frame_3.setStyleSheet("image: url(:/button/img/buttom/多云_cloudy.svg);")
        elif _nowWeather == '小雨':
            self.frame_3.setStyleSheet("image: url(:/button/img/buttom/小雨_light-rain.svg);")
        elif _nowWeather == '小雪':
            self.frame_3.setStyleSheet("image: url(:/button/img/buttom/雪花_snowflake.svg);")
        elif _nowWeather == '大雪':
            self.frame_3.setStyleSheet("image: url(:/button/img/buttom/雪花_snowflake.svg);")
        elif _nowWeather == '雷雨':
            self.frame_3.setStyleSheet("image: url(:/button/img/buttom/雷雨_thunderstorm-one.svg);")
        elif _nowWeather == '大雾':
            self.frame_3.setStyleSheet("image: url(:/button/img/buttom/大雾_fog.svg);")

        # 温度范围
        _lowerTemp = r.json()['results'][0]['daily'][2]['low']
        _higherTemp = r.json()['results'][0]['daily'][2]['high']
        self.label_16.setText(f'{_nowWeather}')
        # 温度范围
        self.label_6.setText(f'{_lowerTemp}°C - {_higherTemp}°C')

        '''初始化数据库'''
        self.conn = sqlite3.connect("static/data.db")
        # 使用cursor
        self.cursor = self.conn.cursor()
        #新建数据表
        # 城市,门店,日期,库存,在租,出租率,总收入
        sql = '''CREATE TABLE IF NOT EXISTS `dataSet`(
        `id` TEXT NOT NULL,
        `Name` TEXT NOT NULL,
        `times` TEXT NOT NULL,
        `things` TEXT NOT NULL);'''
        self.cursor.execute(sql)
        self.conn.commit()

        self.lineEdit.setEnabled(False)
        self.lineEdit_2.setEnabled(False)
        self.lineEdit_3.setEnabled(False)
        self.lineEdit_6.setEnabled(False)
        self.lineEdit_4.setEnabled(False)
        self.lineEdit_7.setEnabled(False)
        self.lineEdit_5.setEnabled(False)
        self.lineEdit_8.setEnabled(False)
        #注册
        self.pushButton_4.clicked.connect(self.register)
        #登录
        self.pushButton_3.clicked.connect(self.login)
        #编辑
        self.pushButton_2.clicked.connect(self.bianji)
        #确认
        self.pushButton.clicked.connect(self.getData)

    def bianji(self):
        self.lineEdit.setEnabled(True)
        self.lineEdit_2.setEnabled(True)
        self.lineEdit_3.setEnabled(True)
        self.lineEdit_6.setEnabled(True)
        self.lineEdit_4.setEnabled(True)
        self.lineEdit_7.setEnabled(True)
        self.lineEdit_5.setEnabled(True)
        self.lineEdit_8.setEnabled(True)

    def getData(self):
        self.lineEdit.setEnabled(False)
        self.lineEdit_2.setEnabled(False)
        self.lineEdit_3.setEnabled(False)
        self.lineEdit_6.setEnabled(False)
        self.lineEdit_4.setEnabled(False)
        self.lineEdit_7.setEnabled(False)
        self.lineEdit_5.setEnabled(False)
        self.lineEdit_8.setEnabled(False)

        #保存数据
        # 更新语句
        sql = '''UPDATE dataSet SET times = ?, things = ? WHERE Name = ? AND id =?'''

        self.cursor.execute(sql, (self.lineEdit.text(),self.lineEdit_2.text(),self._name,"1"))
        self.cursor.execute(sql, (self.lineEdit_3.text(),self.lineEdit_6.text(),self._name,"2"))
        self.cursor.execute(sql, (self.lineEdit_4.text(),self.lineEdit_7.text(),self._name,"3"))
        self.cursor.execute(sql, (self.lineEdit_5.text(),self.lineEdit_8.text(),self._name,"4"))


        self.conn.commit()
        QMessageBox.information(self,"提示","成功",QMessageBox.Yes)

    def register(self):
        self._win = registerWindow(self.conn,self.cursor)
        self._win.d.connect(self.refrsh)
        self._win.show()

    def refrsh(self):
        self._win.close()
        QMessageBox.information(self,"提示","注册成功",QMessageBox.Yes)


    def login(self):
        if not os.path.exists('weight.yml'):
            QMessageBox.warning(self,"警告","未注册人脸",QMessageBox.Yes)
        else:
            self._lo = loginWindow()
            self._lo.d.connect(self.refrshResult)
            self._lo.show()


    def refrshResult(self,_name):
        self._name = _name
        QMessageBox.information(self,"提示",f"{_name}登录成功",QMessageBox.Yes)
        self._lo.close()
        #刷新页面
        # 查询语句
        sql = '''SELECT * FROM dataSet WHERE Name = ?'''
        # 执行查询
        self.cursor.execute(sql, (_name,))
        _datas = self.cursor.fetchall()
        self.conn.commit()

        for value in _datas:
            if value[0] == "1":
                self.lineEdit.setText(value[-2])
                self.lineEdit_2.setText(value[-1])
            elif value[0] == "2":
                self.lineEdit_3.setText(value[-2])
                self.lineEdit_6.setText(value[-1])
            elif value[0] == "3":
                self.lineEdit_4.setText(value[-2])
                self.lineEdit_7.setText(value[-1])
            elif value[0] == "4":
                self.lineEdit_5.setText(value[-2])
                self.lineEdit_8.setText(value[-1])
        self.lineEdit.setEnabled(False)
        self.lineEdit_2.setEnabled(False)
        self.lineEdit_3.setEnabled(False)
        self.lineEdit_6.setEnabled(False)
        self.lineEdit_4.setEnabled(False)
        self.lineEdit_7.setEnabled(False)
        self.lineEdit_5.setEnabled(False)
        self.lineEdit_8.setEnabled(False)






if __name__ == "__main__":
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    #创建QApplication 固定写法
    app = QApplication(sys.argv)
    # 实例化界面
    window = MainWindow()
    #显示界面
    window.show()
    #阻塞，固定写法
    sys.exit(app.exec_())