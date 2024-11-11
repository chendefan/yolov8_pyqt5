import sys
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap, QImage
import cv2
from ultralytics import YOLO
from yolo import Ui_MainWindow


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        try:
            super().__init__()
            self.setupUi(self)
            self.model = YOLO("yolov8n.pt")
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.update_frame)
 
            self.cap = None
            self.is_detection_active = False
            self.current_frame = None
 
            # 各个按钮绑定功能
            self.picture_detect.clicked.connect(self.load_picture)
            self.video_detect.clicked.connect(self.load_video)
            self.camera_detect.clicked.connect(self.load_camera)
            self.start_detect.clicked.connect(self.start_detection)
            self.stop_detect.clicked.connect(self.stop_detection)
            self.pause_detect.clicked.connect(self.pause_detection)
 
        except Exception as e:
            print(e)


    def load_picture(self):
        try:
            fileName, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Image Files (*.jpg *.png)")
            self.is_detection_active = False
 
            if fileName:
                if self.timer.isActive():
                    self.timer.stop()
                if self.cap:
                    self.cap.release()
                    self.cap = None
 
                self.current_frame = cv2.imread(fileName)
                self.display_image(self.current_frame, self.original_image)
                results = self.model.predict(self.current_frame)
                self.detected_frame = results[0].plot()  # 获取检测结果的帧并保存
                self.display_image(self.detected_frame, self.detected_image)
        except Exception as e:
            print(e)
 
    def load_video(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Video Files (*.mp4 *.avi)")
        if fileName:
            if self.cap:
                self.cap.release()
                self.cap = None
 
            self.cap = cv2.VideoCapture(fileName)
 
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame = frame.copy()
                    self.display_image(frame, self.original_image)
                    self.display_image(frame, self.detected_image)
                else:
                    QtWidgets.QMessageBox.warning(self, 'Error', '无法读取视频文件的第一帧。')
 
    def load_camera(self):
        self.is_detection_active = False
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(0)
        self.timer.start(20)
 
    def update_frame(self):
        if self.cap:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame.copy()
                self.display_image(frame, self.original_image)
 
                if self.is_detection_active:
                    results = self.model.predict(frame)
                    self.detected_frame = results[0].plot()  # 获取检测结果的帧并保存
                    self.display_image(self.detected_frame, self.detected_image)
 
    def start_detection(self):
        if self.cap and not self.cap.isOpened():
            self.cap.open(self.fileName)
        if self.cap and not self.timer.isActive():
            self.timer.start(20)
        self.is_detection_active = True
    
    def pause_detection(self):
            self.is_detection_active = False
            if self.timer.isActive():
                self.timer.stop()

    def display_image(self, frame, target_label):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = frame.shape
        step = channel * width
        qImg = QImage(frame.data, width, height, step, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qImg)
        scaled_pixmap = pixmap.scaled(target_label.size(), QtCore.Qt.KeepAspectRatio)
        target_label.setPixmap(scaled_pixmap)
 
 
    def stop_detection(self):
        self.is_detection_active = False
 
        if self.timer.isActive():
            self.timer.stop()
 
        if self.cap:
            self.cap.release()
            self.cap = None
 
        self.clear_display(self.original_image)
        self.clear_display(self.detected_image)
 
    def clear_display(self, target_label):
        target_label.clear()
        target_label.setText('')
 
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())