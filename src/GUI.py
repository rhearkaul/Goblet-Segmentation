import shutil
import tkinter as tk
from datetime import datetime
from tkinter import filedialog

import cv2
from PIL import ImageTk, Image
import numpy as np
import os
from sam2 import sam_main
from metrics import get_prop, analyze_properties

from watershed.watershed import (
    STAIN_VECTORS,
    INTENSITY_THRESHOLDS,
    SIZE_THRESHOLDS,
    generate_centroid,
)

class ImageViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Viewer")
        self.opened_image = None

        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Set the window size to 75% of the screen resolution
        window_width = int(screen_width * 0.75)
        window_height = int(screen_height * 0.75)

        self.geometry(f"{window_width}x{window_height}")
        self.resizable(True, True)  # Allow window resizing

        self.opened_image = None
        self.box_select_mode = False
        self.point_select_mode = False
        self.points = []
        self.boxes = []
        self.point_ids = []
        self.box_ids = []
        self.brush_size = 10  # Default brush size

        self.sam_model_size = "H"
        self.sam_weights_path = "sam_vit_h_4b8939.pth"

        self.watershed_settings = {
            "stain_vector": 0,
            "equalization_bins": 5,
            "intensity_thresh": INTENSITY_THRESHOLDS[0],
            "size_thresh": SIZE_THRESHOLDS[0],
            "max_aspect_ratio": 2.5,
            "min_solidity": 0.55,
        }

        self.create_widgets(window_width, window_height)
        self.create_menubar()

        self.rect_id = None

        self.canvas.mask_images = []

        self.masks = []
        self.mask_files = []
        self.masks_dir = ""
        self.image_name = ""

        self.manual_mask_mode = None

    def create_widgets(self, window_width, window_height):

        toolbar_frame = tk.Frame(self, bg="gray")
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)

        select_toolbar_frame = tk.Frame(self, bg="lightgray")
        select_toolbar_frame.pack(side=tk.TOP, fill=tk.X)

        self.box_select_button = tk.Button(select_toolbar_frame, text="Box Select", command=self.toggle_box_select_mode)
        self.box_select_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.point_select_button = tk.Button(select_toolbar_frame, text="Point Select", command=self.toggle_point_select_mode)
        self.point_select_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.manual_mask_button = tk.Button(select_toolbar_frame, text="Manual Mask",
                                            command=self.toggle_manual_mask_mode)
        self.manual_mask_button.pack(side=tk.LEFT, padx=5, pady=5)

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

        delete_button = tk.Button(self.annotation_window_frame, text="Delete Annotation", command=self.delete_selected_annotation)
        delete_button.pack(side=tk.BOTTOM, padx=5, pady=5)

        # save_button = tk.Button(self.annotation_window_frame, text="Save Annotations", command=self.save_annotations)
        # save_button.pack(side=tk.BOTTOM, padx=5, pady=5)

        unselect_button = tk.Button(self.annotation_window_frame, text="Unselect", command=self.unselect_annotation)
        unselect_button.pack(side=tk.BOTTOM, padx=5, pady=5)

        self.image_viewer_frame = tk.Frame(self, bg="white", width=image_display_width, height=image_display_height)
        self.image_viewer_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        self.image_viewer_frame.pack_propagate(False)

        self.canvas = tk.Canvas(self.image_viewer_frame)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

    def update_brush_size(self, value):
        self.brush_size = int(value)
    def create_menubar(self):
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        menu1 = tk.Menu(menubar, tearoff=0)
        menu2 = tk.Menu(menubar, tearoff=0)
        menu3 = tk.Menu(menubar, tearoff=0)
        menu4 = tk.Menu(menubar, tearoff=0)
        menu5 = tk.Menu(menubar, tearoff=0)

        menu1.add_command(label="Open Image", command=self.open_image)

        menu2.add_command(label="Placeholder 2")
        menu3.add_command(label="Watershed Settings", command=self.show_watershed_settings)
        menu3.add_command(label="Run Watershed", command=self.run_watershed)

        menu5.add_command(label="Run Analysis", command=self.run_analysis)


        menu4 = tk.Menu(menubar, tearoff=0)
        menu4.add_command(label="Run SAM with Current Annotation", command=self.run_sam_with_current_annotation)
        menu4.add_command(label="SAM Settings", command=self.show_sam_settings)


        menubar.add_cascade(label="File", menu=menu1)
        menubar.add_cascade(label="View", menu=menu2)
        menubar.add_cascade(label="WaterShed", menu=menu3)
        menubar.add_cascade(label="SAM", menu=menu4)
        menubar.add_cascade(label="Metric", menu=menu5)

    def show_watershed_settings(self):
        watershed_settings_window = tk.Toplevel(self)
        watershed_settings_window.title("Watershed Settings")

        # Create stain vector input
        stain_vector_label = tk.Label(watershed_settings_window, text="Stain Vector:")
        stain_vector_label.pack()
        stain_vector_var = tk.IntVar()
        stain_vector_var.set(0)  # Set the initial value
        stain_vector_entry = tk.Entry(watershed_settings_window, textvariable=stain_vector_var)
        stain_vector_entry.pack()

        # Create equalization bins input
        equalization_bins_label = tk.Label(watershed_settings_window, text="Equalization Bins:")
        equalization_bins_label.pack()
        equalization_bins_var = tk.IntVar()
        equalization_bins_var.set(5)  # Set the initial value
        equalization_bins_entry = tk.Entry(watershed_settings_window, textvariable=equalization_bins_var)
        equalization_bins_entry.pack()

        # Create intensity threshold input
        intensity_thresh_label = tk.Label(watershed_settings_window, text="Intensity Threshold:")
        intensity_thresh_label.pack()
        intensity_thresh_var = tk.StringVar()
        intensity_thresh_var.set(",".join(map(str, INTENSITY_THRESHOLDS[0])))  # Set the initial value
        intensity_thresh_entry = tk.Entry(watershed_settings_window, textvariable=intensity_thresh_var)
        intensity_thresh_entry.pack()

        # Create size threshold input
        size_thresh_label = tk.Label(watershed_settings_window, text="Size Threshold:")
        size_thresh_label.pack()
        size_thresh_var = tk.StringVar()
        size_thresh_var.set(",".join(map(str, SIZE_THRESHOLDS[0])))  # Set the initial value
        size_thresh_entry = tk.Entry(watershed_settings_window, textvariable=size_thresh_var)
        size_thresh_entry.pack()

        # Create max aspect ratio input
        max_aspect_ratio_label = tk.Label(watershed_settings_window, text="Max Aspect Ratio:")
        max_aspect_ratio_label.pack()
        max_aspect_ratio_var = tk.DoubleVar()
        max_aspect_ratio_var.set(2.5)  # Set the initial value
        max_aspect_ratio_entry = tk.Entry(watershed_settings_window, textvariable=max_aspect_ratio_var)
        max_aspect_ratio_entry.pack()

        # Create min solidity input
        min_solidity_label = tk.Label(watershed_settings_window, text="Min Solidity:")
        min_solidity_label.pack()
        min_solidity_var = tk.DoubleVar()
        min_solidity_var.set(0.55)  # Set the initial value
        min_solidity_entry = tk.Entry(watershed_settings_window, textvariable=min_solidity_var)
        min_solidity_entry.pack()

        # Create save button
        def save_settings():
            stain_vector_index = stain_vector_var.get()
            if stain_vector_index in STAIN_VECTORS:
                self.watershed_settings = {
                    "stain_vector": stain_vector_index,
                    "equalization_bins": equalization_bins_var.get(),
                    "intensity_thresh": tuple(map(float, intensity_thresh_var.get().split(","))),
                    "size_thresh": tuple(map(int, size_thresh_var.get().split(","))),
                    "max_aspect_ratio": max_aspect_ratio_var.get(),
                    "min_solidity": min_solidity_var.get(),
                }
            else:
                print(f"Invalid stain vector index: {stain_vector_index}")

        save_button = tk.Button(watershed_settings_window, text="Save", command=save_settings)
        save_button.pack(pady=10)

        watershed_settings_window.mainloop()

    def run_watershed(self):
        if self.opened_image:
            image = cv2.imread(self.image_path)
            stain_vector = STAIN_VECTORS[self.watershed_settings["stain_vector"]]
            centroid_coords, deconv_img, segmented_img, distances = generate_centroid(
                image,
                stain_vector,
                self.watershed_settings["equalization_bins"],
                self.watershed_settings["intensity_thresh"],
                self.watershed_settings["size_thresh"],
                self.watershed_settings["max_aspect_ratio"],
                self.watershed_settings["min_solidity"],
            )

            # Clear existing point annotations
            self.clear_points_and_boxes()

            # Add centroid coordinates as point annotations
            for coord in centroid_coords:
                x, y = coord
                oval_id = self.canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill='red')
                self.points.append((x, y))
                self.point_ids.append(oval_id)

            self.update_annotation_listbox()
        else:
            print("No image opened. Please open an image first.")

    def show_sam_settings(self):
        sam_settings_window = tk.Toplevel(self)
        sam_settings_window.title("SAM Settings")

        # Create model size input
        model_size_label = tk.Label(sam_settings_window, text="Model Size:")
        model_size_label.pack()
        model_size_var = tk.StringVar()
        model_size_var.set(self.sam_model_size)  # Set the initial value from the instance variable
        model_size_entry = tk.Entry(sam_settings_window, textvariable=model_size_var)
        model_size_entry.pack()

        # Create path to weights input
        weights_path_label = tk.Label(sam_settings_window, text="Path to Weights:")
        weights_path_label.pack()
        weights_path_var = tk.StringVar()
        weights_path_var.set(self.sam_weights_path)  # Set the initial value from the instance variable
        weights_path_entry = tk.Entry(sam_settings_window, textvariable=weights_path_var)
        weights_path_entry.pack()

        # Create save button
        def save_settings():
            self.sam_model_size = model_size_var.get()
            self.sam_weights_path = weights_path_var.get()
            model_size_var.set(self.sam_model_size)  # Update the displayed value
            weights_path_var.set(self.sam_weights_path)  # Update the displayed value

        save_button = tk.Button(sam_settings_window, text="Save", command=save_settings)
        save_button.pack(pady=10)

        sam_settings_window.mainloop()

    def open_image(self):
        image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.tif;*.tiff")])
        if image_path:
            self.image_path = image_path
            self.image_name = os.path.splitext(os.path.basename(image_path))[0]
            self.create_unique_image_folder()
            self.copy_image_to_folder()
            image = cv2.imread(os.path.join(self.image_folder, os.path.basename(image_path)))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            self.opened_image = image
            photo = ImageTk.PhotoImage(image)
            self.canvas.image = photo
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.clear_annotations()
            print(f"Image opened: {image_path}")
        else:
            self.opened_image = None
            self.canvas.delete("all")
            print("No image selected.")

    def create_unique_image_folder(self):
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        self.image_folder = f"image_masks/{self.image_name}_{timestamp}"
        os.makedirs(self.image_folder, exist_ok=True)

    def copy_image_to_folder(self):
        shutil.copy2(self.image_path, self.image_folder)

    def load_masks(self):
        self.masks = []
        self.mask_files = []
        all_files = [f for f in os.listdir(self.image_folder) if f.endswith(".png")]
        mask_files = [f for f in all_files if
                      not f.startswith(self.image_name) and not f.startswith("predicted_") and not f.startswith(
                          "mask_")]
        mask_files = sorted(mask_files, key=lambda x: os.path.getmtime(os.path.join(self.image_folder, x)))
        for mask_file in mask_files:
            mask_path = os.path.join(self.image_folder, mask_file)
            mask = cv2.imread(mask_path, 0)
            self.masks.append(mask)
            self.mask_files.append(mask_file)
        self.display_masks()
        self.update_annotation_listbox()

    def create_loading_screen(self):
        self.loading_screen = tk.Toplevel(self)
        self.loading_screen.title("Loading")
        self.loading_screen.geometry("200x100")
        self.loading_screen.resizable(False, False)

        label = tk.Label(self.loading_screen, text="Running SAM, Please Wait.")
        label.pack(pady=20)

    def load_existing_masks(self):
        self.masks_dir = f"output_masks/{self.image_name}"
        self.masks = []
        self.mask_files = []

        # Load SAM-generated masks
        if os.path.exists(self.masks_dir):
            for root, dirs, files in os.walk(self.masks_dir):
                for file in files:
                    if file.startswith("mask_"):
                        mask_path = os.path.join(root, file)
                        mask = cv2.imread(mask_path, 0)
                        self.masks.append(mask)
                        self.mask_files.append(file)

        # Load manual masks
        for root, dirs, files in os.walk(self.image_folder):
            for file in files:
                if file.startswith("mask_"):
                    mask_path = os.path.join(root, file)
                    mask = cv2.imread(mask_path, 0)
                    self.masks.append(mask)
                    self.mask_files.append(file)

        self.display_masks()
        self.update_annotation_listbox()
        self.canvas.delete("mask")  # Remove existing mask items from the canvas

        for i, mask in enumerate(self.masks):
            mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
            mask_rgba[..., :3] = [30, 144, 255]  # Blue color
            mask_rgba[..., 3] = (mask > 0).astype(np.uint8) * 128  # Set alpha channel based on mask
            mask_image = Image.fromarray(mask_rgba, mode='RGBA')
            mask_photo = ImageTk.PhotoImage(mask_image)
            self.canvas.mask_images.append(mask_photo)  # Keep a reference to the mask photo
            self.canvas.create_image(0, 0, anchor=tk.NW, image=mask_photo, tags=f"mask_{i}")

    def toggle_manual_mask_mode(self):
        if self.opened_image:
            self.manual_mask_mode = not self.manual_mask_mode
            self.box_select_mode = False
            self.point_select_mode = False
            if self.manual_mask_mode:
                self.manual_mask_button.configure(bg="lightblue")
                self.box_select_button.configure(bg="lightgray")
                self.point_select_button.configure(bg="lightgray")
                # Clean up box select mode
                if self.rect_id:
                    self.canvas.delete(self.rect_id)
                    self.rect_id = None
                # Clean up point select mode
                for point_id in self.point_ids:
                    self.canvas.delete(point_id)
                self.points = []
                self.point_ids = []
                self.update_annotation_listbox()
            else:
                self.manual_mask_button.configure(bg="lightgray")
        else:
            print("No image opened. Please open an image first.")
    def display_masks(self):
        for i, mask in enumerate(self.masks):
            binary_mask = mask > 0
            mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
            mask_rgba[..., :3] = [30, 144, 255]  # Blue color
            mask_rgba[..., 3] = binary_mask.astype(np.uint8) * 128  # Set alpha channel based on mask

            mask_image = Image.fromarray(mask_rgba, mode='RGBA')
            mask_photo = ImageTk.PhotoImage(mask_image)
            self.canvas.mask_images.append(mask_photo)  # Keep a reference to the mask photo
            self.canvas.create_image(0, 0, anchor=tk.NW, image=mask_photo, tags=f"mask_{i}")

    def toggle_box_select_mode(self):
        if self.opened_image:
            self.box_select_mode = not self.box_select_mode
            self.point_select_mode = False
            self.manual_mask_mode = False  # Reset manual mask mode
            if self.box_select_mode:
                self.box_select_button.configure(bg="lightblue")
                self.point_select_button.configure(bg="lightgray")
                self.manual_mask_button.configure(bg="lightgray")  # Reset manual mask button color
                self.canvas.delete("manual_mask")  # Clear manual mask drawing
                self.manual_mask_path = []  # Reset manual mask path
            else:
                self.box_select_button.configure(bg="lightgray")
        else:
            print("No image opened. Please open an image first.")

    def toggle_point_select_mode(self):
        if self.opened_image:
            self.point_select_mode = not self.point_select_mode
            self.box_select_mode = False
            self.manual_mask_mode = False  # Reset manual mask mode
            if self.point_select_mode:
                self.point_select_button.configure(bg="lightblue")
                self.box_select_button.configure(bg="lightgray")
                self.manual_mask_button.configure(bg="lightgray")  # Reset manual mask button color
                self.canvas.delete("manual_mask")  # Clear manual mask drawing
                self.manual_mask_path = []  # Reset manual mask path
            else:
                self.point_select_button.configure(bg="lightgray")
        else:
            print("No image opened. Please open an image first.")

    def on_canvas_click(self, event):
        if self.opened_image:
            if self.manual_mask_mode:
                self.manual_mask_start_x = event.x
                self.manual_mask_start_y = event.y
                self.manual_mask_path = [(event.x, event.y)]
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
        if self.opened_image:
            if self.box_select_mode:
                if self.rect_id:
                    self.canvas.delete(self.rect_id)
                self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, event.x, event.y, outline='red')
            elif self.manual_mask_mode:
                self.canvas.delete("manual_mask")
                self.manual_mask_path.append((event.x, event.y))
                self.canvas.create_line(self.manual_mask_path, fill='red', tags="manual_mask", width=self.brush_size)

    def on_canvas_release(self, event):
        if self.opened_image:
            if self.manual_mask_mode:
                self.manual_mask_path.append((event.x, event.y))
                self.create_manual_mask()
            elif self.box_select_mode:
                x1 = min(self.start_x, event.x)
                y1 = min(self.start_y, event.y)
                x2 = max(self.start_x, event.x)
                y2 = max(self.start_y, event.y)
                self.boxes.append((x1, y1, x2, y2))
                self.box_ids.append(self.rect_id)
                self.rect_id = None  # Reset rect_id after appending it to box_ids
                self.update_annotation_listbox()

    def create_manual_mask(self):
        mask = np.zeros((self.opened_image.height, self.opened_image.width), dtype=np.uint8)
        manual_mask_polygon = np.array(self.manual_mask_path, dtype=np.int32)
        cv2.polylines(mask, [manual_mask_polygon], False, 255, thickness=self.brush_size)

        # Save the manual mask to the image folder with a timestamp
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        mask_index = len(self.mask_files)
        mask_file = f"manual_mask_{timestamp}_{mask_index}.png"
        mask_path = os.path.join(self.image_folder, mask_file)
        cv2.imwrite(mask_path, mask)

        self.masks.append(mask)
        self.mask_files.append(mask_file)
        self.display_masks()
        self.update_annotation_listbox()
        self.manual_mask_path = []
        self.canvas.delete("manual_mask")

    def update_annotation_listbox(self):
        self.annotation_listbox.delete(0, tk.END)
        for mask_file in self.mask_files:
            self.annotation_listbox.insert(tk.END, mask_file)

    def on_annotation_select(self, event):
        selection = self.annotation_listbox.curselection()
        self.highlight_annotations(selection)

        # Highlight the selected mask
        if len(selection) == 1:
            index = selection[0]
            if index >= len(self.points) + len(self.boxes):
                mask_index = index - len(self.points) - len(self.boxes)
                self.highlight_mask(mask_index)
            else:
                self.highlight_mask(-1)  # Clear highlight if no mask is selected
        else:
            self.highlight_mask(-1)  # Clear highlight if multiple items are selected

    def highlight_mask(self, mask_index):
        self.canvas.delete("highlight")  # Remove any existing highlight
        if mask_index >= 0 and mask_index < len(self.masks):
            mask = self.masks[mask_index]
            highlight_mask = np.zeros_like(mask, dtype=np.uint8)
            highlight_mask[mask > 0] = 255

            # Create a transparent highlight overlay
            highlight_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
            highlight_rgba[..., :3] = [255, 255, 0]  # Yellow color
            highlight_rgba[..., 3] = (highlight_mask > 0).astype(
                np.uint8) * 128  # Set alpha channel based on highlight mask

            highlight_image = Image.fromarray(highlight_rgba, mode='RGBA')
            highlight_photo = ImageTk.PhotoImage(highlight_image)
            self.canvas.highlight_image = highlight_photo  # Keep a reference to the highlight photo
            self.canvas.create_image(0, 0, anchor=tk.NW, image=highlight_photo, tags="highlight")

    def clear_mask_highlight(self):
        for i in range(len(self.mask_files)):
            self.canvas.itemconfig(f"mask_{i}", state=tk.HIDDEN)

    def unselect_annotation(self):
        self.annotation_listbox.selection_clear(0, tk.END)
        self.highlight_mask(-1)  # Clear the highlight
        # Display all masks with the original blue color
        for i in range(len(self.mask_files)):
            self.canvas.itemconfig(f"mask_{i}", state=tk.NORMAL)
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
        for i, mask in enumerate(self.masks):
            if mask[y, x] > 0:
                self.annotation_listbox.selection_clear(0, tk.END)
                self.annotation_listbox.selection_set(len(self.points) + len(self.boxes) + i)
                self.highlight_mask(i)
                return

        self.highlight_mask(-1)  # Clear highlight if no mask is clicked

    def delete_selected_annotation(self):
        selection = self.annotation_listbox.curselection()
        if selection:
            indices = list(selection)
            indices.sort(reverse=True)  # Delete from the end to avoid shifting indices
            for index in indices:
                if index < len(self.points):
                    # Deleting a point annotation
                    del self.points[index]
                    point_id = self.point_ids.pop(index)
                    self.canvas.delete(point_id)
                elif index < len(self.points) + len(self.boxes):
                    # Deleting a box annotation
                    box_index = index - len(self.points)
                    if box_index < len(self.boxes):
                        del self.boxes[box_index]
                        box_id = self.box_ids.pop(box_index)
                        self.canvas.delete(box_id)
                else:
                    # Deleting a mask annotation
                    mask_index = index - len(self.points) - len(self.boxes)
                    if mask_index < len(self.mask_files):
                        mask_file = self.mask_files[mask_index]
                        mask_file_path = os.path.join(self.image_folder, mask_file)
                        if os.path.exists(mask_file_path):
                            os.remove(mask_file_path)  # Delete the mask file
                        del self.masks[mask_index]
                        del self.mask_files[mask_index]
                        # Remove the mask from the canvas using its tag
                        self.canvas.delete(f"mask_{mask_index}")
                        # Adjust the tags of the remaining masks to reflect the new indexing
                        self.retag_masks_after_deletion(mask_index)

            self.update_annotation_listbox()
            self.clear_mask_highlight()

            # Reload all masks in the folder
            self.load_masks()

    def retag_masks_after_deletion(self, deleted_mask_index):
        for i in range(deleted_mask_index, len(self.masks)):
            self.canvas.itemconfig(f"mask_{i + 1}", tags=f"mask_{i}")

    def save_current_annotations(self):
        if self.opened_image:
            default_filename = 'current_annotations.npz'
            points_array = np.array(self.points)
            boxes_array = np.array(self.boxes)
            np.savez(default_filename, points=points_array, boxes=boxes_array, image_path=self.image_path)
            print(f"Annotations saved successfully as {default_filename}.")
        else:
            print("No image opened. Please open an image before saving annotations.")

    def clear_annotations(self):
        self.points = []
        self.boxes = []
        self.update_annotation_listbox()
        self.canvas.delete("annotation")
        if self.opened_image is not None:
            photo = ImageTk.PhotoImage(self.opened_image)
            self.canvas.image = photo
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)

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

    def run_sam_with_current_annotation(self):
        self.save_current_annotations()
        path_to_weights = self.sam_weights_path

        self.create_loading_screen()
        self.update()

        output_dir = sam_main(path_to_weights, annotations_filename='current_annotations.npz',
                              image_folder=self.image_folder, model_size=self.sam_model_size)
        print("SAM function executed with current annotation.")

        self.loading_screen.destroy()

        # Save SAM-generated masks with a timestamp
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
        mask_files = [f for f in os.listdir(output_dir) if f.startswith("mask_")]
        for i, mask_file in enumerate(mask_files):
            src_path = os.path.join(output_dir, mask_file)
            dst_path = os.path.join(self.image_folder, f"sam_mask_{timestamp}_{i}.png")
            shutil.copy2(src_path, dst_path)

        # Remove the original "mask_*.png" files
        for mask_file in mask_files:
            os.remove(os.path.join(output_dir, mask_file))

        self.load_masks()
        self.clear_points_and_boxes()

    def clear_points_and_boxes(self):
        for point_id in self.point_ids:
            self.canvas.delete(point_id)
        for box_id in self.box_ids:
            self.canvas.delete(box_id)
        self.points = []
        self.boxes = []
        self.point_ids = []
        self.box_ids = []
        self.update_annotation_listbox()

    def display_image_with_masks(self, masks_dir):
        image_path = os.path.join(masks_dir, 'predicted_image.png')
        image, masks, mask_files = self.load_image_with_masks(image_path, masks_dir)

        self.opened_image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(self.opened_image)
        self.canvas.delete("all")
        self.canvas.image = photo
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)

        for i, mask in enumerate(masks):
            mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
            mask_rgba[..., :3] = [30, 144, 255]  # Blue color
            mask_rgba[..., 3] = (mask > 0).astype(np.uint8) * 128  # Set alpha channel based on mask

            mask_image = Image.fromarray(mask_rgba, mode='RGBA')
            mask_photo = ImageTk.PhotoImage(mask_image)
            self.canvas.mask_images.append(mask_photo)  # Keep a reference to the mask photo
            self.canvas.create_image(0, 0, anchor=tk.NW, image=mask_photo, tags=f"mask_{i}")

        self.masks = masks
        self.mask_files = mask_files
        self.masks_dir = masks_dir
        self.update_annotation_listbox()

    def load_image_with_masks(self, image_path, masks_dir):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_files = [mask for mask in os.listdir(masks_dir) if mask.startswith('mask_')]
        masks = [cv2.imread(os.path.join(masks_dir, mask_file), 0) for mask_file in mask_files]
        return image, masks, mask_files

    def run_analysis(self):
        if not self.masks:
            print("No masks found. Please run SAM first.")
            return

        binary_masks = [mask > 0 for mask in self.masks]

        res_x, res_y = self.opened_image.info.get("resolution", (None, None))
        if not res_x or not res_y:
            res_x = res_y = 1  # Set a default resolution if not available

        results = []
        for binary_mask in binary_masks:
            prop_df = get_prop(binary_mask)
            if not prop_df.empty:
                result = analyze_properties(prop_df, res_x)
                results.append(result)

        print("Analysis Results:")
        for i, result in enumerate(results):
            print(f"Mask {i + 1}:")
            print(result)
            print()


if __name__ == "__main__":
    app = ImageViewer()
    app.mainloop()

