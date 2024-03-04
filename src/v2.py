import os
from tkinter import Tk, Canvas, Frame, BOTH, Menu, Toplevel, Listbox, END
from tkinter import filedialog
import PIL
from PIL import Image, ImageTk
import numpy as np
from tkinter import Button
from sam2 import sam_main
import src.watershed.watershed

class ImageViewer(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        self.initUI()
        self.annotation_mode = None
        self.start_x = None
        self.start_y = None
        self.rect = None
        self.points = []
        self.boxes = []
        self.point_ids = []
        self.box_ids = []

        self.slice_rect = None
        self.slice_coords = None

    def initUI(self):
        self.parent.title("Image Viewer")
        self.pack(fill=BOTH, expand=1)
        self.canvas = Canvas(self)
        self.canvas.pack(fill=BOTH, expand=1)
        self.createMenuBar()

    def createMenuBar(self):
        menubar = Menu(self.parent)
        self.parent.config(menu=menubar)

        fileMenu = Menu(menubar)
        fileMenu.add_command(label="Open", command=self.onOpen)
        menubar.add_cascade(label="File", menu=fileMenu)

        annotatorMenu = Menu(menubar)
        annotatorMenu.add_command(label="Point select", command=lambda: self.setAnnotationMode('point'))
        annotatorMenu.add_command(label="Box select", command=lambda: self.setAnnotationMode('box'))
        annotatorMenu.add_command(label="Show annotations", command=self.showAnnotations)
        menubar.add_cascade(label="Annotator", menu=annotatorMenu)

        sliceMenu = Menu(menubar)
        sliceMenu.add_command(label="Slice Area Select", command=self.setSliceMode)
        sliceMenu.add_command(label="Confirm Slice", command=self.confirmSlice)
        menubar.add_cascade(label="Slice", menu=sliceMenu)

        samMenu = Menu(menubar)
        samMenu.add_command(label="Run SAM", command=self.runSAM)
        menubar.add_cascade(label="SAM", menu=samMenu)

    def runSAM(self):
        path_to_weights = "sam_vit_h_4b8939.pth"
        sam_main(path_to_weights)
        print("SAM function executed.")

    def onOpen(self):
        ftypes = [('Image files', '*.jpg *.jpeg *.png *.gif *.bmp')]
        dlg = filedialog.Open(self, filetypes=ftypes)
        fl = dlg.show()

        if fl != '':
            self.image_path = fl  # Save the image path
            self.loadImage(fl)

    def loadImage(self, fl):
        self.image = Image.open(fl)
        self.photo = ImageTk.PhotoImage(self.image)
        self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
        self.canvas.config(scrollregion=self.canvas.bbox("all"))

    def setAnnotationMode(self, mode):
        self.annotation_mode = mode
        if mode == 'point':
            self.canvas.bind("<Button-1>", self.onCanvasClick)
        elif mode == 'box':
            self.canvas.bind("<Button-1>", self.onStartBox)
            self.canvas.bind("<B1-Motion>", self.onDrag)
            self.canvas.bind("<ButtonRelease-1>", self.onRelease)

    def onCanvasClick(self, event):
        if self.annotation_mode == 'point':
            oval_id = self.canvas.create_oval(event.x - 2, event.y - 2, event.x + 2, event.y + 2, fill='red')
            self.points.append((event.x, event.y))
            self.point_ids.append(oval_id)

    def onStartBox(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def onDrag(self, event):
        if self.annotation_mode == 'box' and self.start_x is not None and self.start_y is not None:
            if self.rect:
                self.canvas.delete(self.rect)
            self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline='green')

    def onRelease(self, event):
        if self.annotation_mode == 'box' and self.start_x is not None and self.start_y is not None:
            self.boxes.append((self.start_x, self.start_y, event.x, event.y))
            self.box_ids.append(self.rect)
            self.rect = None
            self.start_x = None
            self.start_y = None

    def showAnnotations(self):
        self.annotatorWindow = Toplevel(self)
        self.annotatorWindow.title("Annotations")
        self.annotatorWindow.geometry("300x400")

        self.lb = Listbox(self.annotatorWindow)
        self.lb.pack(fill=BOTH, expand=True)

        for point in self.points:
            self.lb.insert(END, f"Point: {point}")

        for box in self.boxes:
            self.lb.insert(END, f"Box: {box}")

        delete_button = Button(self.annotatorWindow, text="Delete Selected", command=self.deleteSelected)
        delete_button.pack(side='top')

        save_button = Button(self.annotatorWindow, text="Save Annotations", command=self.saveAnnotations)
        save_button.pack(side='top')



    def deleteSelected(self):
        selected = self.lb.curselection()
        for index in reversed(selected):
            if index < len(self.points):
                self.canvas.delete(self.point_ids[index])
                del self.points[index]
                del self.point_ids[index]
            else:
                box_index = index - len(self.points)
                self.canvas.delete(self.box_ids[box_index])
                del self.boxes[box_index]
                del self.box_ids[box_index]
            self.lb.delete(index)

    def saveAnnotations(self):
        if hasattr(self, 'image_path'):
            points_array = np.array(self.points)
            boxes_array = np.array(self.boxes)
            np.savez('annotations.npz', points=points_array, boxes=boxes_array, image_path=self.image_path)
            print("Annotations and image path saved.")
        else:
            print("No image opened.")

    def setSliceMode(self):
        self.annotation_mode = 'slice'
        self.canvas.bind("<Button-1>", self.onStartSlice)
        self.canvas.bind("<B1-Motion>", self.onDragSlice)
        self.canvas.bind("<ButtonRelease-1>", self.onReleaseSlice)

    def onStartSlice(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def onDragSlice(self, event):
        if self.slice_rect:
            self.canvas.delete(self.slice_rect)
        self.slice_rect = self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline='blue')

    def onReleaseSlice(self, event):
        self.slice_coords = (self.start_x, self.start_y, event.x, event.y)

    def confirmSlice(self):
        if self.slice_coords:
            cropped_image = self.image.crop(self.slice_coords)

            base, ext = os.path.splitext(self.image_path)
            sliced_image_path = f"{base}_sliced{ext}"

            cropped_image.save(sliced_image_path)

            self.image_path = sliced_image_path
            self.image = cropped_image

            self.photo = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, image=self.photo, anchor='nw')
            self.canvas.config(scrollregion=self.canvas.bbox("all"))

            if self.slice_rect:
                self.canvas.delete(self.slice_rect)
            self.slice_rect = None
            self.slice_coords = None

def main():
    root = Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    app_width = int(screen_width * 0.75)
    app_height = int(screen_height * 0.75)
    app = ImageViewer(root)
    root.geometry(f"{app_width}x{app_height}+100+100")
    root.mainloop()

if __name__ == '__main__':
    main()

# used to load
# def loadAnnotations(filename='annotations.npz'):
#     data = np.load(filename)
#     input_points = data['points']
#     input_boxes = data['boxes']
#     print("Loaded points:", input_points)
#     print("Loaded boxes:", input_boxes)
#     return input_points, input_boxes