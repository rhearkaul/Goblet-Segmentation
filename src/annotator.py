import os
import csv
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QFileDialog, QListWidget, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QMessageBox, QToolBar)
from PyQt5.QtGui import QPixmap, QPainter, QPen, QBrush, QColor
from PyQt5.QtCore import Qt, QRect


class ImageWindow(QWidget):
    def __init__(self, imagePath):
        super().__init__()
        self.imagePath = imagePath
        self.marks = []
        self.boxes = []
        self.mode = 'Point'
        self.tempBox = None

        self.imageLabel = QLabel()
        self.originalPixmap = QPixmap(imagePath)
        self.imageLabel.setPixmap(self.originalPixmap)

        self.setWindowTitle("Image Viewer")
        self.setWindowFlags(Qt.WindowCloseButtonHint)

        self.initUI()
        self.show()

    def initUI(self):
        layout = QVBoxLayout()
        self.toolbar = self.initToolbar()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.imageLabel)

        self.setLayout(layout)

    def initToolbar(self):
        toolbar = QToolBar("Mode Toolbar")
        pointModeAction = toolbar.addAction("Point Mode")
        boxModeAction = toolbar.addAction("Box Mode")
        pointModeAction.triggered.connect(lambda: self.setMode('Point'))
        boxModeAction.triggered.connect(lambda: self.setMode('Box'))
        return toolbar

    def setMode(self, mode):
        self.mode = mode

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = self.imageLabel.mapFromParent(event.pos())
            if self.mode == 'Point':
                self.mark(pos)
                MainWindow.instance().addMark(pos)
            elif self.mode == 'Box' and self.imageLabel.rect().contains(pos):
                self.tempBox = [pos, pos]

    def mouseMoveEvent(self, event):
        if self.mode == 'Box' and self.tempBox:
            pos = self.imageLabel.mapFromParent(event.pos())
            if self.imageLabel.rect().contains(pos):
                self.tempBox[1] = pos
                self.update()

    def mouseReleaseEvent(self, event):
        if self.mode == 'Box' and self.tempBox:
            pos = self.imageLabel.mapFromParent(event.pos())
            if self.imageLabel.rect().contains(pos):
                self.boxes.append(QRect(self.tempBox[0], pos))
                MainWindow.instance().addBox(self.boxes[-1])
                self.tempBox = None
                self.update()

    def mark(self, position):
        self.marks.append(position)
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        pixmap = self.originalPixmap.copy()
        painter = QPainter(pixmap)
        painter.setPen(QPen(Qt.red, 10))
        painter.setBrush(QBrush(QColor(255, 0, 0, 50)))
        for pos in self.marks:
            painter.drawPoint(pos)
        for box in self.boxes:
            painter.drawRect(box)
        if self.tempBox:
            painter.drawRect(QRect(self.tempBox[0], self.tempBox[1]))
        painter.end()
        self.imageLabel.setPixmap(pixmap)


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

    def addBox(self, box):
        self.markListWidget.addItem(
            f"Box: {box.topLeft().x()}, {box.topLeft().y()} to {box.bottomRight().x()}, {box.bottomRight().y()}")

    def deleteMark(self):
        selectedItems = self.markListWidget.selectedItems()
        if selectedItems and self.imageWindow:
            text = selectedItems[0].text()
            if text.startswith("X:"):
                # Deleting a point mark
                index = self.markListWidget.row(selectedItems[0])
                del self.imageWindow.marks[index]
            elif text.startswith("Box:"):
                # Deleting a box
                index = self.markListWidget.row(selectedItems[0])
                del self.imageWindow.boxes[index]
            self.markListWidget.takeItem(index)
            self.imageWindow.update()

    def saveMarks(self):
        if self.imageWindow and (self.imageWindow.marks or self.imageWindow.boxes):
            filename, _ = QFileDialog.getSaveFileName(self, "Save File", "", "CSV files (*.csv)")
            if filename:
                with open(filename, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['Type', 'X1', 'Y1', 'X2', 'Y2'])
                    for mark in self.imageWindow.marks:
                        writer.writerow(['Point', mark.x(), mark.y(), '', ''])
                    for box in self.imageWindow.boxes:
                        writer.writerow(
                            ['Box', box.topLeft().x(), box.topLeft().y(), box.bottomRight().x(), box.bottomRight().y()])
                QMessageBox.information(self, "Save Marks", "Marks and boxes saved successfully.")
        else:
            QMessageBox.warning(self, "Save Marks", "No marks or boxes to save.")

if __name__ == '__main__':
    app = QApplication([])
    mainWindow = MainWindow()
    mainWindow.show()
    app.exec_()
