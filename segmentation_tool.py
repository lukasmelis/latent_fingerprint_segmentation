import sys, cv2, os
import modules.kmeans as kmeans 
import modules.mask_rcnn as mrcnn
from PyQt5 import QtWidgets, QtCore
from PyQt5.uic import loadUi
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *

class ChooseWindow(QDialog):
    def __init__(self):
        super(ChooseWindow, self).__init__()
        loadUi("design/volba.ui", self)
        self.start_btn.clicked.connect(self.start)        
        self.browse_btn.clicked.connect(self.browse_files)
        self.paths = ''
       

    def start(self):
        global seg_type
        seg_type = ""
        if self.kmeans_rb.isChecked():
            seg_type = self.kmeans_rb.text() 
        elif self.mask_rb.isChecked():
            seg_type = self.mask_rb.text() 
        else:
            seg_type = self.com_rb.text()

        if seg_type != self.com_rb.text():    
            segWind = SegmentationWindow(self.paths, widget.currentIndex())
            widget.addWidget(segWind)
            widget.setCurrentIndex(widget.currentIndex()+1)
            widget.setFixedWidth(1060)
            widget.setFixedHeight(570)
        else:
            compWind = ComparisonWindow(self.paths, widget.currentIndex())
            widget.addWidget(compWind)
            widget.setCurrentIndex(widget.currentIndex()+1)
            widget.setFixedWidth(1500)
            widget.setFixedHeight(570)


    def browse_files(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.DirectoryOnly)
        dlg.setAcceptMode(QFileDialog.AcceptOpen)
        if dlg.exec_() == QDialog.Accepted:
            dir_path = dlg.selectedFiles()[0]
            self.paths = kmeans.create_image_paths(dir_path) 

class SegmentationWindow(QDialog):
    def __init__(self, paths, prev_wid):
        super(SegmentationWindow, self).__init__()
        loadUi("design/segmentacia.ui", self)
        self.prev_wid = prev_wid
        self.image = None 
        self.image_mrcnn = None
        self.pos = 0
        self.paths = paths
        self.disply_width = 400
        self.display_height = 500
        self.next_btn.clicked.connect(self.next)
        self.back_btn.clicked.connect(self.back)
        self.save_btn.clicked.connect(self.save)
        self.display()

       
    def get_img_data(self):
        print("SEG_TYPE", seg_type)
        if seg_type == "K-Means":
            self.image = kmeans.main(self.paths[self.pos])
        elif seg_type == "Mask R-CNN":
            mrcnn.main(self.paths[self.pos])
            self.image = "tmp.png"
            self.image_mrcnn = cv2.imread(self.image)
         

    def display(self):
        self.counter.setText(str((self.pos + 1)) + '/' + str(len(self.paths)))
        self.get_img_data()
        self.in_lb.setPixmap(QPixmap(self.paths[self.pos]).
                scaled(self.disply_width, self.display_height,  Qt.KeepAspectRatio, Qt.SmoothTransformation))
        if seg_type == "Mask R-CNN":
            self.out_lb.setPixmap(QPixmap(self.image).
                    scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            os.remove(self.image)
        else:
            rgb_image = cv2.cvtColor(self.image, cv2.COLOR_GRAY2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            self.out_lb.setPixmap(QPixmap.fromImage(p))
        

    def save(self):
        sv_dlg = QFileDialog()
        sv_dlg.setAcceptMode(QFileDialog.AcceptSave) 
        sv_dlg.setNameFilter("*.jpg, *.png, *.jpeg")
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = sv_dlg.getSaveFileName(self, filter = ("*.jpg *.png *.jpeg"), options=options)
            
        if filename:
            if seg_type == "Mask R-CNN":
                cv2.imwrite(filename, self.image_mrcnn)
            else:
                cv2.imwrite(filename, self.image)


    def next(self): 
        if self.pos + 1 < len(self.paths):
            self.pos += 1
            self.display()


    def back(self):
        widget.removeWidget(self)
        widget.setCurrentIndex(self.prev_wid)
        widget.setFixedWidth(360)
        widget.setFixedHeight(290)


class ComparisonWindow(QDialog):
    def __init__(self, paths, prev_wid):
        super(ComparisonWindow, self).__init__()
        loadUi("design/porovnanie.ui", self)
        self.prev_wid = prev_wid
        self.paths = paths
        self.tmp = "tmp.png"
        self.image_kmeans = None 
        self.image_mrcnn = None
        self.pos = 0
        self.disply_width = 400
        self.display_height = 500
        self.next_btn.clicked.connect(self.next)
        self.back_btn.clicked.connect(self.back)
        self.display()


    def get_img_data(self):
        self.image_kmeans = kmeans.main(self.paths[self.pos])
        mrcnn.main(self.paths[self.pos])
        self.image_mrcnn = cv2.imread(self.tmp)


    def display(self):
        self.counter.setText(str((self.pos + 1)) + '/' + str(len(self.paths)))
        self.get_img_data()
        self.in_lb.setPixmap(QPixmap(self.paths[self.pos]).
            scaled(self.disply_width, self.display_height,  Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.mrcnn_lb.setPixmap(QPixmap(self.tmp).
                scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        os.remove(self.tmp)
        rgb_image = cv2.cvtColor(self.image_kmeans, cv2.COLOR_GRAY2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.kmeans_lb.setPixmap(QPixmap.fromImage(p))


    def next(self): 
        if self.pos + 1 < len(self.paths):
            self.pos += 1
            self.display()
    

    def back(self):
        widget.removeWidget(self)
        widget.setCurrentIndex(self.prev_wid)
        widget.setFixedWidth(360)
        widget.setFixedHeight(290)



if __name__ == '__main__': 
    app=QApplication(["Segmentačný nástroj latentných odtlačkov"])
    mainwindow=ChooseWindow()

    widget=QtWidgets.QStackedWidget()
    widget.addWidget(mainwindow)
    widget.setFixedWidth(360)
    widget.setFixedHeight(290)
    widget.show()
    sys.exit(app.exec_())



