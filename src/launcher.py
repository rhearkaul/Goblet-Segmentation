import sys
import os
import csv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QListWidget, QLabel, QFileDialog, QGraphicsView, QGraphicsScene,
                             QGraphicsPixmapItem, QGraphicsEllipseItem)
from PyQt5.QtGui import QPixmap, QPainter, QPen
from PyQt5.QtCore import Qt, QPointF

class ImageViewer(QGraphicsView):
    def __init__(self, imagePath):
        super().__init__()
        self.imagePath = imagePath
        self.marks = []
        self.markListWidget = None
        self.initUI()

    def initUI(self):
        self.scene = QGraphicsScene(self)
        self.pixmapItem = QGraphicsPixmapItem(QPixmap(self.imagePath))
        self.scene.addItem(self.pixmapItem)
        self.setScene(self.scene)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            position = self.mapToScene(event.pos())
            self.addMark(position.x(), position.y())

    def addMark(self, x, y):
        mark = QGraphicsEllipseItem(x - 5, y - 5, 10, 10)
        mark.setBrush(Qt.red)
        self.scene.addItem(mark)
        self.marks.append((x, y))
        if self.markListWidget:
            self.markListWidget.addItem(f"X: {x}, Y: {y}")

    def setMarkListWidget(self, markListWidget):
        self.markListWidget = markListWidget

class ImageSelector(QMainWindow):
    def __init__(self, folderPath):
        super().__init__()
        self.folderPath = folderPath
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Selector')
        self.layout = QVBoxLayout()

        self.imageListWidget = QListWidget()
        self.populateImageList()

        openButton = QPushButton("Open")
        openButton.clicked.connect(self.openImage)

        self.layout.addWidget(self.imageListWidget)
        self.layout.addWidget(openButton)

        centralWidget = QWidget()
        centralWidget.setLayout(self.layout)
        self.setCentralWidget(centralWidget)

    def populateImageList(self):
        for filename in os.listdir(self.folderPath):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                self.imageListWidget.addItem(filename)

    def openImage(self):
        selectedItems = self.imageListWidget.selectedItems()
        if selectedItems:
            filename = selectedItems[0].text()
            imagePath = os.path.join(self.folderPath, filename)
            self.imageViewer = ImageViewerWindow(imagePath)
            self.imageViewer.show()

class ImageViewerWindow(QMainWindow):
    def __init__(self, imagePath):
        super().__init__()
        self.imagePath = imagePath
        self.marks = []
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Image Viewer')
        self.viewer = ImageViewer(self.imagePath)

        self.layout = QVBoxLayout()

        self.markListWidget = QListWidget()
        self.viewer.setMarkListWidget(self.markListWidget)

        deleteMarkButton = QPushButton("Delete Mark")
        deleteMarkButton.clicked.connect(self.deleteMark)

        saveButton = QPushButton("Save")
        saveButton.clicked.connect(self.saveMarks)

        self.layout.addWidget(self.viewer)
        self.layout.addWidget(self.markListWidget)
        self.layout.addWidget(deleteMarkButton)
        self.layout.addWidget(saveButton)

        centralWidget = QWidget()
        centralWidget.setLayout(self.layout)
        self.setCentralWidget(centralWidget)
        self.resize(self.viewer.pixmapItem.pixmap().width(), self.viewer.pixmapItem.pixmap().height())

    def deleteMark(self):
        selectedItems = self.markListWidget.selectedItems()
        if selectedItems:
            for item in selectedItems:
                index = self.markListWidget.row(item)
                del self.viewer.marks[index]
                self.markListWidget.takeItem(index)

    def saveMarks(self):
        filePath, _ = QFileDialog.getSaveFileName(self, "Save File", "", "CSV files (*.csv)")
        if filePath:
            with open(filePath, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['X', 'Y'])
                for mark in self.viewer.marks:
                    writer.writerow(mark)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # Replace 'your/folder/path' with the actual folder path obtained from Group Box 3
    folderPath = 'your/folder/path'
    selector = ImageSelector(folderPath)
    selector.show()
    sys.exit(app.exec_())




