from PyQt5.QtWidgets import (QApplication, QMainWindow, QGroupBox, QVBoxLayout, QHBoxLayout,
                             QPushButton, QSpinBox, QLabel, QTextEdit, QFileDialog, QWidget)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('PyQt5 GUI')
        self.setGeometry(100, 100, 640, 360)  # x, y, width, height

        # Dictionary to store folder paths for each group box
        self.folderPaths = {}

        self.initUI()

    def initUI(self):
        widget = self.createCentralWidget()
        self.setCentralWidget(widget)

    def createCentralWidget(self):
        mainLayout = QVBoxLayout()

        # Creating group boxes
        groupBox1 = self.createGroupBox("WaterShed settings", True)
        groupBox2 = self.createGroupBox("WaterShed settings", False)
        groupBox3 = self.createGroupBox("Annotator settings", True)
        groupBox4 = self.createGroupBox("Analyze settings", True)

        # Adding group boxes to the main layout
        mainLayout.addWidget(groupBox1)
        mainLayout.addWidget(groupBox2)
        mainLayout.addWidget(groupBox3)
        mainLayout.addWidget(groupBox4)

        centralWidget = QWidget()
        centralWidget.setLayout(mainLayout)
        return centralWidget

    def createGroupBox(self, title, includeFolderSelect):
        groupBox = QGroupBox(title)
        layout = QVBoxLayout()

        # Parameters layout
        for i in range(1, 4):
            paramLayout = QHBoxLayout()
            paramLayout.addWidget(QLabel(f"Parameter {i}:"))
            paramLayout.addWidget(QSpinBox())
            layout.addLayout(paramLayout)

        # Folder select and path display
        if includeFolderSelect:
            folderSelectButton = QPushButton("Folder Select")
            folderSelectButton.clicked.connect(lambda: self.selectFolder(groupBox))

            pathTextEdit = QTextEdit()
            pathTextEdit.setFixedHeight(40)
            layout.addWidget(folderSelectButton)
            layout.addWidget(pathTextEdit)

        # Start button
        startButton = QPushButton("Start" if title != "WaterShed settings" else "Start WaterShed")
        layout.addWidget(startButton)

        groupBox.setLayout(layout)
        return groupBox

    def selectFolder(self, groupBox):
        folderPath = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folderPath:  # Check if a folder was selected
            # Find the QTextEdit in the group box to update the path
            for child in groupBox.findChildren(QTextEdit):
                child.setText(folderPath)
                break

            # Store the folder path in the dictionary using the group box as the key
            self.folderPaths[groupBox] = folderPath

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()





