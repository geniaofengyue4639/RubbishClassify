import sys
from MainTest_UI import *
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from garbage_eval import evalvation
from module_eval import module_eval
from label_collect import labelpath_cut
import numpy as np
import cv2 as cv


class Main(QMainWindow, Ui_window):
    """重写主窗体类"""

    def __init__(self, parent=None):
        super(Main, self).__init__(parent)
        self.setupUi(self)  # 初始化窗体显示
        # 设置在label中自适应显示图片
        self.label_Input.setScaledContents(True)
        self.label_Input.setStyleSheet("QLabel{background-color:rgb(0,0,0);}")  # 初始黑化图像显示区域
        self.img = None
        self.img_path = None

    def img_show(self, label, img):
        """图片在对应label中显示"""
        if img.shape[-1] == 3:
            qimage = QImage(img.data.tobytes(), img.shape[1], img.shape[0], img.shape[1] * 3,
                            QImage.Format_RGB888).rgbSwapped()
        else:
            qimage = QImage(img.data.tobytes(), img.shape[1], img.shape[0], img.shape[1], QImage.Format_Indexed8)
        label.setPixmap(QPixmap.fromImage(qimage))

    def img_open(self):
        """"点击读入图片"""
        image_name, image_type = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;*.png;*.jpeg;*.bmp")
        self.img_path = image_name
        np_image = cv.imdecode(np.fromfile(image_name, dtype=np.uint8), -1)
        self.img = np_image
        qt_image = QImage(np_image.data.tobytes(), np_image.shape[1], np_image.shape[0], np_image.shape[1] * 3,
                          QImage.Format_RGB888).rgbSwapped()  # 以8位灰度的方式，将numpy_image转为QImage
        self.label_Input.setPixmap(QPixmap.fromImage(qt_image))
        self.img_lable()

    def img_lable(self):
        """图片标签显示"""
        mes = labelpath_cut(self.img_path)
        self.textBrowser_realabel.append(mes)
        self.cursot = self.textBrowser_realabel.textCursor()
        self.textBrowser_realabel.moveCursor(self.cursot.End)

    def text_printf(self, mes):
        """"结果显示"""
        self.textBrowser_AlexNet.append(mes[0])  # 在指定的区域显示提示信息
        self.textBrowser_GoogLenet.append(mes[1])  # 在指定的区域显示提示信息
        self.textBrowser_VGG.append(mes[2])
        self.textBrowser_ResNet.append(mes[3])  # 在指定的区域显示提示信息
        self.cursot = self.textBrowser_AlexNet.textCursor()
        self.textBrowser_AlexNet.moveCursor(self.cursot.End)
        self.cursot = self.textBrowser_GoogLenet.textCursor()
        self.textBrowser_GoogLenet.moveCursor(self.cursot.End)
        self.cursot = self.textBrowser_VGG.textCursor()
        self.textBrowser_VGG.moveCursor(self.cursot.End)
        self.cursot = self.textBrowser_ResNet.textCursor()
        self.textBrowser_ResNet.moveCursor(self.cursot.End)

    def singleimage_eval(self):
        """"点击进行单张图片预测"""
        img = self.img
        mes = evalvation(img)
        self.text_printf(mes)

    def module_acc_eval(self):
        """"点击模型预测"""
        mes = module_eval()
        self.text_printf(mes)

    def closeEvent(self, event):
        """重写QMainWindow类的closeEvent事件，自定义关闭过程"""
        reply = QMessageBox.question(self, "提示", "确认退出?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:  # 判断返回值，如果点击的是Yes，接受关闭事件，否则忽略关闭事件
            event.accept()
        else:
            event.ignore()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Main()
    win.show()
    sys.exit(app.exec_())
