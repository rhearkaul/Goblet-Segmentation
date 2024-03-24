import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import os

class ImageViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Viewer")

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Set the window size to 75% of the screen resolution
        window_width = int(screen_width * 0.75)
        window_height = int(screen_height * 0.75)

        self.geometry(f"{window_width}x{window_height}")
        self.resizable(False, False)  # Disable window resizing

        self.opened_image = None
        self.box_select_mode = False
        self.point_select_mode = False
        self.points = []
        self.boxes = []
        self.point_ids = []
        self.box_ids = []

        self.create_widgets(window_width, window_height)

        self.rect_id = None

    def create_widgets(self, window_width, window_height):

        toolbar_frame = tk.Frame(self, bg="gray")
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)

        open_button = tk.Button(toolbar_frame, text="Open", command=self.open_image)
        open_button.pack(side=tk.LEFT, padx=5, pady=5)

        input_button = tk.Button(toolbar_frame, text="Step 1: Input")
        input_button.pack(side=tk.LEFT, padx=5, pady=5)

        select_button = tk.Button(toolbar_frame, text="Step 2: Select")
        select_button.pack(side=tk.LEFT, padx=5, pady=5)

        watershed_button = tk.Button(toolbar_frame, text="Step 3: Watershed")
        watershed_button.pack(side=tk.LEFT, padx=5, pady=5)

        segmentation_button = tk.Button(toolbar_frame, text="Step 4: Segmentation")
        segmentation_button.pack(side=tk.LEFT, padx=5, pady=5)

        analysis_button = tk.Button(toolbar_frame, text="Step 5: Analysis")
        analysis_button.pack(side=tk.LEFT, padx=5, pady=5)

        select_toolbar_frame = tk.Frame(self, bg="lightgray")
        select_toolbar_frame.pack(side=tk.TOP, fill=tk.X)

        self.box_select_button = tk.Button(select_toolbar_frame, text="Box Select", command=self.toggle_box_select_mode)
        self.box_select_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.point_select_button = tk.Button(select_toolbar_frame, text="Point Select", command=self.toggle_point_select_mode)
        self.point_select_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Calculate the sizes of function window and image display area
        function_window_width = int(window_width * 0.2)
        function_window_height = window_height - 80
        image_display_width = window_width - function_window_width
        image_display_height = window_height - 80

        self.annotation_window_frame = tk.Frame(self, bg="white", width=function_window_width, height=function_window_height)
        self.annotation_window_frame.pack(side=tk.LEFT, fill=tk.BOTH)
        self.annotation_window_frame.pack_propagate(False)  # Disable frame size adjustment

        self.annotation_listbox = tk.Listbox(self.annotation_window_frame, selectmode=tk.EXTENDED)
        self.annotation_listbox.pack(fill=tk.BOTH, expand=True)
        self.annotation_listbox.bind("<<ListboxSelect>>", self.on_annotation_select)

        delete_button = tk.Button(self.annotation_window_frame, text="Delete Selected Annotation", command=self.delete_selected_annotation)
        delete_button.pack(side=tk.BOTTOM, padx=5, pady=5)

        save_button = tk.Button(self.annotation_window_frame, text="Save Annotations", command=self.save_annotations)
        save_button.pack(side=tk.BOTTOM, padx=5, pady=5)

        self.image_viewer_frame = tk.Frame(self, bg="white", width=image_display_width, height=image_display_height)
        self.image_viewer_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        self.image_viewer_frame.pack_propagate(False)

        self.canvas = tk.Canvas(self.image_viewer_frame)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.opened_image = Image.open(file_path)
            self.opened_image.thumbnail((600, 400))
            photo = ImageTk.PhotoImage(self.opened_image)
            self.canvas.delete("all")
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo
            self.clear_annotations()

    def toggle_box_select_mode(self):
        if self.opened_image:
            self.box_select_mode = not self.box_select_mode
            self.point_select_mode = False
            if self.box_select_mode:
                self.box_select_button.configure(bg="lightblue")
                self.point_select_button.configure(bg="lightgray")
            else:
                self.box_select_button.configure(bg="lightgray")
        else:
            print("No image opened. Please open an image first.")

    def toggle_point_select_mode(self):
        if self.opened_image:
            self.point_select_mode = not self.point_select_mode
            self.box_select_mode = False
            if self.point_select_mode:
                self.point_select_button.configure(bg="lightblue")
                self.box_select_button.configure(bg="lightgray")
            else:
                self.point_select_button.configure(bg="lightgray")
        else:
            print("No image opened. Please open an image first.")

    def on_canvas_click(self, event):
        if self.opened_image:
            if self.box_select_mode:
                self.start_x = event.x
                self.start_y = event.y
            elif self.point_select_mode:
                x = event.x
                y = event.y
                oval_id = self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill='red')
                self.points.append((x, y))
                self.point_ids.append(oval_id)
                self.update_annotation_listbox()
            else:
                self.check_annotation_click(event.x, event.y)

    def on_canvas_drag(self, event):
        if self.opened_image and self.box_select_mode:
            if self.rect_id:
                self.canvas.delete(self.rect_id)
            self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline='red')

    def on_canvas_release(self, event):
        if self.opened_image and self.box_select_mode:
            x1 = min(self.start_x, event.x)
            y1 = min(self.start_y, event.y)
            x2 = max(self.start_x, event.x)
            y2 = max(self.start_y, event.y)
            self.boxes.append((x1, y1, x2, y2))
            self.box_ids.append(self.rect_id)
            self.update_annotation_listbox()

    def update_annotation_listbox(self):
        self.annotation_listbox.delete(0, tk.END)
        for i, point in enumerate(self.points):
            self.annotation_listbox.insert(tk.END, f"Point {i+1}: ({point[0]}, {point[1]})")
        for i, box in enumerate(self.boxes):
            self.annotation_listbox.insert(tk.END, f"Box {i+1}: ({box[0]}, {box[1]}, {box[2]}, {box[3]})")

    def on_annotation_select(self, event):
        selection = self.annotation_listbox.curselection()
        self.highlight_annotations(selection)

    def highlight_annotations(self, selection):
        self.canvas.delete("highlight")
        for index in selection:
            if index < len(self.points):
                point = self.points[index]
                self.canvas.create_oval(point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4, outline='yellow', tags="highlight")
            else:
                box_index = index - len(self.points)
                if box_index < len(self.boxes):
                    box = self.boxes[box_index]
                    self.canvas.create_rectangle(box[0], box[1], box[2], box[3], outline='yellow', tags="highlight")

    def check_annotation_click(self, x, y):
        for i, box in enumerate(self.boxes):
            if box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                self.annotation_listbox.selection_clear(0, tk.END)
                self.annotation_listbox.selection_set(len(self.points) + i)
                self.highlight_annotations([len(self.points) + i])
                return
        for i, point in enumerate(self.points):
            if abs(point[0] - x) <= 2 and abs(point[1] - y) <= 2:
                self.annotation_listbox.selection_clear(0, tk.END)
                self.annotation_listbox.selection_set(i)
                self.highlight_annotations([i])
                return

    def delete_selected_annotation(self):
        selection = self.annotation_listbox.curselection()
        if selection:
            indices = list(selection)
            indices.sort(reverse=True)
            for index in indices:
                if index < len(self.points):
                    del self.points[index]
                    point_id = self.point_ids.pop(index)
                    self.canvas.delete(point_id)
                else:
                    box_index = index - len(self.points)
                    if box_index < len(self.boxes):
                        del self.boxes[box_index]
                        box_id = self.box_ids.pop(box_index)
                        self.canvas.delete(box_id)
            self.update_annotation_listbox()
            self.canvas.delete("highlight")

    def save_annotations(self):
        annotation_file = filedialog.asksaveasfilename(defaultextension=".npz", filetypes=[("NumPy Files", "*.npz")])
        if annotation_file:
            points_array = np.array(self.points)
            boxes_array = np.array(self.boxes)
            np.savez(annotation_file, points=points_array, boxes=boxes_array)
            print(f"Annotations saved to {annotation_file}")

    def clear_annotations(self):
        self.points = []
        self.boxes = []
        self.point_ids = []
        self.box_ids = []
        self.update_annotation_listbox()
        self.canvas.delete("all")
        if self.opened_image:
            photo = ImageTk.PhotoImage(self.opened_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.canvas.image = photo

    def loadAnnotations(self, annotation_file):
        # Load annotations from a given file
        if os.path.exists(annotation_file):
            data = np.load(annotation_file, allow_pickle=True)
            loaded_points = data['points']
            loaded_boxes = data['boxes']

            # Clear existing annotations
            for point_id in self.point_ids:
                self.canvas.delete(point_id)
            for box_id in self.box_ids:
                self.canvas.delete(box_id)

            # Clear the lists
            self.points.clear()
            self.boxes.clear()
            self.point_ids.clear()
            self.box_ids.clear()

            # Load new annotations
            for point in loaded_points:
                oval_id = self.canvas.create_oval(point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2, fill='red')
                self.points.append((point[0], point[1]))
                self.point_ids.append(oval_id)

            for box in loaded_boxes:
                rect_id = self.canvas.create_rectangle(box[0], box[1], box[2], box[3], outline='green')
                self.boxes.append((box[0], box[1], box[2], box[3]))
                self.box_ids.append(rect_id)
        else:
            print(f"Annotation file {annotation_file} not found.")

if __name__ == "__main__":
    app = ImageViewer()
    app.mainloop()

