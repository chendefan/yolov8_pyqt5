import sys
from PyQt5 import QtWidgets,QtCore
app = QtWidgets.QApplication(sys.argv)
widget = QtWidgets.QWidget()
widget.resize(360,360)
widget.setWindowTitle("First")
widget.show()
sys.exit(app.exec_())
