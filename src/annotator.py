import os
import csv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog, QListWidget, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QMessageBox)
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt

class ImageWindow(QLabel):
    def __init__(self, imagePath):
        super().__init__()
        self.imagePath = imagePath
        self.marks = []
        self.setPixmap(QPixmap(imagePath))
        self.setWindowTitle("Image Viewer")
        self.setWindowFlags(Qt.WindowCloseButtonHint)
        self.show()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.mark(event.pos())
            MainWindow.instance().addMark(event.pos())

    def mark(self, position):
        self.marks.append(position)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(QPen(Qt.red, 10))
        for pos in self.marks:
            painter.drawPoint(pos)

class MainWindow(QMainWindow):
    _instance = None

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Selector")
        self.setGeometry(100, 100, 640, 480)
        MainWindow._instance = self

        self.folderPath = ""
        self.imageListWidget = QListWidget()
        self.markListWidget = QListWidget()
        self.imageWindow = None

        self.initUI()

    @staticmethod
    def instance():
        return MainWindow._instance

    def initUI(self):
        selectFolderBtn = QPushButton("Select Folder")
        selectFolderBtn.clicked.connect(self.selectFolder)

        openImageBtn = QPushButton("Open")
        openImageBtn.clicked.connect(self.openImage)

        deleteMarkBtn = QPushButton("Delete Mark")
        deleteMarkBtn.clicked.connect(self.deleteMark)

        saveMarksBtn = QPushButton("Save")
        saveMarksBtn.clicked.connect(self.saveMarks)

        layout = QVBoxLayout()
        layout.addWidget(selectFolderBtn)
        layout.addWidget(self.imageListWidget)
        layout.addWidget(openImageBtn)
        layout.addWidget(self.markListWidget)
        layout.addWidget(deleteMarkBtn)
        layout.addWidget(saveMarksBtn)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def selectFolder(self):
        folderPath = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folderPath:
            self.folderPath = folderPath
            self.populateImageList()

    def populateImageList(self):
        self.imageListWidget.clear()
        for filename in os.listdir(self.folderPath):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                self.imageListWidget.addItem(filename)

    def openImage(self):
        selectedItems = self.imageListWidget.selectedItems()
        if selectedItems:
            imagePath = os.path.join(self.folderPath, selectedItems[0].text())
            self.imageWindow = ImageWindow(imagePath)

    def addMark(self, position):
        self.markListWidget.addItem(f"X: {position.x()}, Y: {position.y()}")

    def deleteMark(self):
        selectedItems = self.markListWidget.selectedItems()
        if selectedItems and self.imageWindow:
            index = self.markListWidget.row(selectedItems[0])
            del self.imageWindow.marks[index]
            self.markListWidget.takeItem(index)
            self.imageWindow.update()

    def saveMarks(self):
        if self.imageWindow and self.imageWindow.marks:
            filename, _ = QFileDialog.getSaveFileName(self, "Save File", "", "CSV files (*.csv)")
            if filename:
                with open(filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['X', 'Y'])
                    for mark in self.imageWindow.marks:
                        writer.writerow([mark.x(), mark.y()])
                QMessageBox.information(self, "Save Marks", "Marks saved successfully.")
        else:
            QMessageBox.warning(self, "Save Marks", "No marks to save.")

if __name__ == '__main__':
    app = QApplication([])
    mainWindow = MainWindow()
    mainWindow.show()
    app.exec_()
