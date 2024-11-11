# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'yolo.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(900, 900)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(40, 120, 160, 181))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.picture_detect = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.picture_detect.setObjectName("picture_detect")
        self.verticalLayout.addWidget(self.picture_detect)
        self.video_detect = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.video_detect.setObjectName("video_detect")
        self.verticalLayout.addWidget(self.video_detect)
        self.camera_detect = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.camera_detect.setObjectName("camera_detect")
        self.verticalLayout.addWidget(self.camera_detect)
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(40, 520, 160, 191))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.start_detect = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.start_detect.setObjectName("start_detect")
        self.verticalLayout_2.addWidget(self.start_detect)
        self.stop_detect = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.stop_detect.setObjectName("stop_detect")
        self.verticalLayout_2.addWidget(self.stop_detect)
        self.pause_detect = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.pause_detect.setObjectName("pause_detect")
        self.verticalLayout_2.addWidget(self.pause_detect)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(460, 20, 81, 20))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(460, 430, 101, 20))
        self.label_2.setObjectName("label_2")
        self.original_image = QtWidgets.QLabel(self.centralwidget)
        self.original_image.setGeometry(QtCore.QRect(250, 50, 500, 350))
        self.original_image.setMaximumSize(QtCore.QSize(600, 600))
        self.original_image.setStyleSheet("QLabel {\n"
"    background-color: white;\n"
"    color: white;\n"
"}\n"
"")
        self.original_image.setObjectName("original_image")
        self.detected_image = QtWidgets.QLabel(self.centralwidget)
        self.detected_image.setGeometry(QtCore.QRect(250, 470, 500, 350))
        self.detected_image.setMaximumSize(QtCore.QSize(600, 600))
        self.detected_image.setStyleSheet("QLabel {\n"
"    background-color: white;\n"
"    color: white;\n"
"}\n"
"")
        self.detected_image.setObjectName("detected_image")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 900, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.picture_detect.setText(_translate("MainWindow", "图片检测"))
        self.video_detect.setText(_translate("MainWindow", "视频检测"))
        self.camera_detect.setText(_translate("MainWindow", "摄像头检测"))
        self.start_detect.setText(_translate("MainWindow", "开始"))
        self.stop_detect.setText(_translate("MainWindow", "暂停"))
        self.pause_detect.setText(_translate("MainWindow", "结束本次检测"))
        self.label.setText(_translate("MainWindow", "原始输入"))
        self.label_2.setText(_translate("MainWindow", "输出结果"))
        self.original_image.setText(_translate("MainWindow", "初始图片"))
        self.detected_image.setText(_translate("MainWindow", "检测后的图片"))
