"""The main package contains the graphic user interface code and handles
the user inputs through the util packages form the other packages.

Author: Ankang Luo 
Co-author: Alvin Hendricks
"""

import logging
import os
import shutil
import tkinter as tk
from datetime import datetime
from tkinter import filedialog
from xml.etree import ElementTree as ET

import cv2
import numpy as np
import pandas as pd
from aicspylibczi import CziFile
from PIL import Image, ImageTk

from metrics import _MEASURED_PROPS, analyze_properties, detect_outliers, get_prop
from sam.sam import SAModel, SAModelType
from sam.util import sam_main
from watershed.watershed import (
    INTENSITY_THRESHOLDS,
    SIZE_THRESHOLDS,
    STAIN_VECTORS,
    generate_centroid,
)

logging.basicConfig(level=logging.INFO)


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

        self.sam_model_size = "L"
        self.sam_weights_path = "sam_vit_l_0b3195.pth"

        self.watershed_settings = {
            "stain_vector": 0,
            "equalization_bins": 5,
            "intensity_thresh": INTENSITY_THRESHOLDS[0],
            "size_thresh": SIZE_THRESHOLDS[2],
            "max_aspect_ratio": 2.5,
            "min_solidity": 0.55,
            "min_area": 300,
            # "dist_thresh": 30,
        }

        self.drag_mode = False
        self.drag_start_x = None
        self.drag_start_y = None
        self.create_widgets(window_width, window_height)
        self.create_menubar()

        self.rect_id = None

        self.canvas.mask_images = []
        self.canvas.mask_highlight_images = []

        self.masks = []
        self.mask_files = []
        self.masks_dir = ""
        self.image_name = ""

        self.manual_mask_mode = None
        self.drag_coefficient_x = 0
        self.drag_coefficient_y = 0

        self.multi_select_mode = True

        # Minimap
        self.minimap_window = None
        self.minimap_canvas = None
        self.minimap_image = None
        self.minimap_rect = None

        self.minimap_drag_coefficient_x = 0
        self.minimap_drag_coefficient_y = 0

        self.sam = None



    def create_widgets(self, window_width, window_height):

        toolbar_frame = tk.Frame(self, bg="gray")
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)

        select_toolbar_frame = tk.Frame(self, bg="lightgray")
        select_toolbar_frame.pack(side=tk.TOP, fill=tk.X)

        self.drag_button = tk.Button(
            select_toolbar_frame, text="Drag", command=self.toggle_drag_mode
        )
        self.drag_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.box_select_button = tk.Button(
            select_toolbar_frame, text="Box Select", command=self.toggle_box_select_mode
        )
        self.box_select_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.point_select_button = tk.Button(
            select_toolbar_frame,
            text="Point Select",
            command=self.toggle_point_select_mode,
        )
        self.point_select_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.manual_mask_button = tk.Button(
            select_toolbar_frame,
            text="Manual Mask",
            command=self.toggle_manual_mask_mode,
        )
        self.manual_mask_button.pack(side=tk.LEFT, padx=5, pady=5)

        minimap_button = tk.Button(
            toolbar_frame, text="Toggle Minimap", command=self.toggle_minimap
        )
        minimap_button.pack(side=tk.LEFT, padx=5, pady=5)

        run_segmentation_all_button = tk.Button(
            toolbar_frame,
            text="Run Segmentation (All)",
            command=self.run_sam_with_current_annotation,
        )
        run_segmentation_all_button.pack(side=tk.LEFT, padx=5, pady=5)

        run_segmentation_selected_button = tk.Button(
            toolbar_frame,
            text="Run Segmentation (Selected)",
            command=self.run_sam_with_selected_annotations,
        )
        run_segmentation_selected_button.pack(side=tk.LEFT, padx=5, pady=5)

        run_analysis_button = tk.Button(
            toolbar_frame, text="Run Analysis", command=self.run_analysis
        )
        run_analysis_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Calculate the sizes of function window and image display area
        function_window_width = int(window_width * 0.2)
        function_window_height = window_height - 80
        image_display_width = window_width - function_window_width
        image_display_height = window_height - 80

        self.annotation_window_frame = tk.Frame(
            self, bg="white", width=function_window_width, height=function_window_height
        )
        self.annotation_window_frame.pack(side=tk.LEFT, fill=tk.BOTH)
        self.annotation_window_frame.pack_propagate(
            False
        )  # Disable frame size adjustment

        self.annotation_listbox = tk.Listbox(
            self.annotation_window_frame, selectmode=tk.EXTENDED
        )
        self.annotation_listbox.pack(fill=tk.BOTH, expand=True)
        self.annotation_listbox.bind("<<ListboxSelect>>", self.on_annotation_select)

        delete_button = tk.Button(
            self.annotation_window_frame,
            text="Delete Annotation",
            command=self.delete_selected_annotation,
        )
        delete_button.pack(side=tk.BOTTOM, padx=5, pady=5)

        unselect_button = tk.Button(
            self.annotation_window_frame,
            text="Unselect",
            command=self.unselect_annotation,
        )
        unselect_button.pack(side=tk.BOTTOM, padx=5, pady=5)

        self.multi_select_mode = True
        self.multi_select_button = tk.Button(
            self.annotation_window_frame,
            text="Multi Select",
            command=self.toggle_multi_select_mode,
        )
        self.multi_select_button.pack(side=tk.BOTTOM, padx=5, pady=5)
        self.multi_select_button.configure(
            bg="lightgray" if not self.multi_select_mode else "lightblue"
        )

        self.image_viewer_frame = tk.Frame(
            self, bg="white", width=image_display_width, height=image_display_height
        )
        self.image_viewer_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        self.image_viewer_frame.pack_propagate(False)

        self.canvas = tk.Canvas(self.image_viewer_frame)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_canvas_release)

    def toggle_minimap(self):
        if self.opened_image:
            if not self.minimap_window or not self.minimap_window.winfo_exists():
                self.create_minimap_window()
            else:
                self.minimap_window.destroy()
                self.minimap_window = None
                self.minimap_canvas = None
                self.minimap_image = None
                self.minimap_rect = None

    def create_minimap_window(self):
        # Calculate the minimap size based on the screen resolution
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        minimap_width = int(screen_width * 0.2)
        minimap_height = int(screen_height * 0.2)

        self.minimap_window = tk.Toplevel(self)
        self.minimap_window.title("Minimap")
        self.minimap_window.geometry(f"{minimap_width}x{minimap_height}")
        self.minimap_window.resizable(False, False)

        self.minimap_canvas = tk.Canvas(
            self.minimap_window, width=minimap_width, height=minimap_height
        )
        self.minimap_canvas.pack()

        # Resize the minimap image to fit the minimap window
        minimap_image = self.opened_image.copy()
        minimap_image = minimap_image.resize(
            (minimap_width, minimap_height), resample=Image.BICUBIC
        )
        self.minimap_image = ImageTk.PhotoImage(minimap_image)
        self.minimap_canvas.create_image(0, 0, anchor=tk.NW, image=self.minimap_image)

        # Calculate the box size and position based on the main canvas size
        canvas_x, canvas_y = self.canvas.winfo_rootx(), self.canvas.winfo_rooty()
        frame_width = self.winfo_width() - canvas_x
        frame_height = self.winfo_height() - canvas_y

        minimap_box_width = int(frame_width * minimap_width / self.opened_image.width)
        minimap_box_height = int(
            frame_height * minimap_height / self.opened_image.height
        )
        minimap_box_x = 0
        minimap_box_y = 0

        self.minimap_rect = self.minimap_canvas.create_rectangle(
            minimap_box_x,
            minimap_box_y,
            minimap_box_x + minimap_box_width,
            minimap_box_y + minimap_box_height,
            outline="red",
        )

        self.minimap_window.bind("<Configure>", self.update_minimap_rect)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)

    def update_minimap_rect(self, event):
        if self.minimap_window and self.minimap_canvas and self.minimap_rect:
            minimap_width = self.minimap_canvas.winfo_width()
            minimap_height = self.minimap_canvas.winfo_height()
            canvas_x, canvas_y = self.canvas.winfo_rootx(), self.canvas.winfo_rooty()
            frame_width = self.winfo_width() - canvas_x
            frame_height = self.winfo_height() - canvas_y

            minimap_box_width = int(
                frame_width * minimap_width / self.opened_image.width
            )
            minimap_box_height = int(
                frame_height * minimap_height / self.opened_image.height
            )

            minimap_box_x = int(
                self.minimap_drag_coefficient_x
                * minimap_width
                / self.opened_image.width
            )
            minimap_box_y = int(
                self.minimap_drag_coefficient_y
                * minimap_height
                / self.opened_image.height
            )

            self.minimap_canvas.coords(
                self.minimap_rect,
                minimap_box_x,
                minimap_box_y,
                minimap_box_x + minimap_box_width,
                minimap_box_y + minimap_box_height,
            )

    def toggle_multi_select_mode(self):
        self.multi_select_mode = not self.multi_select_mode
        if self.multi_select_mode:
            self.annotation_listbox.config(selectmode=tk.EXTENDED)
            self.multi_select_button.configure(
                bg="lightblue"
            )  # Change background color when enabled
        else:
            self.annotation_listbox.config(selectmode=tk.BROWSE)
            self.multi_select_button.configure(
                bg="lightgray"
            )  # Change background color when disabled

    def toggle_drag_mode(self):
        if self.opened_image:
            self.drag_mode = not self.drag_mode
            self.box_select_mode = False
            self.point_select_mode = False
            self.manual_mask_mode = False
            if self.drag_mode:
                self.drag_button.configure(bg="lightblue")
                self.box_select_button.configure(bg="lightgray")
                self.point_select_button.configure(bg="lightgray")
                self.manual_mask_button.configure(bg="lightgray")
                self.canvas.config(cursor="hand2")
            else:
                self.drag_button.configure(bg="lightgray")
                self.canvas.config(cursor="")
        else:
            logging.warning("No image opened. Please open an image first.")

    def update_brush_size(self, value):
        self.brush_size = int(value)

    def create_menubar(self):
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        menu1 = tk.Menu(menubar, tearoff=0)
        menu3 = tk.Menu(menubar, tearoff=0)
        menu4 = tk.Menu(menubar, tearoff=0)
        menu5 = tk.Menu(menubar, tearoff=0)

        menu1.add_command(label="Open Image", command=self.open_image)

        menu3.add_command(
            label="Watershed Settings", command=self.show_watershed_settings
        )
        menu3.add_command(label="Run Watershed", command=self.run_watershed)

        menu5.add_command(label="Run Analysis", command=self.run_analysis)

        menu4 = tk.Menu(menubar, tearoff=0)
        menu4.add_command(label="SAM Settings", command=self.show_sam_settings)
        menu4.add_command(
            label="Segment all annotations",
            command=self.run_sam_with_current_annotation,
        )
        menu4.add_command(
            label="Segment selected annotations",
            command=self.run_sam_with_selected_annotations,
        )

        menubar.add_cascade(label="File", menu=menu1)
        menubar.add_cascade(label="Prompt Generation", menu=menu3)
        menubar.add_cascade(label="Segment", menu=menu4)
        menubar.add_cascade(label="Metrics", menu=menu5)

    def show_watershed_settings(self):
        watershed_settings_window = tk.Toplevel(self)
        watershed_settings_window.title("Watershed Settings")

        warning_label = tk.Label(
            watershed_settings_window,
            text="Watershed is unstable.\nPlease consult manual for more info.",
            fg="red",
        )
        warning_label.pack(pady=10)

        # Create stain vector input
        stain_vector_label = tk.Label(watershed_settings_window, text="Stain Vector:")
        stain_vector_label.pack()
        stain_vector_var = tk.IntVar()
        stain_vector_var.set(0)  # Set the initial value
        stain_vector_entry = tk.Entry(
            watershed_settings_window, textvariable=stain_vector_var
        )
        stain_vector_entry.pack()

        # Create equalization bins input
        equalization_bins_label = tk.Label(
            watershed_settings_window, text="Equalization Bins:"
        )
        equalization_bins_label.pack()
        equalization_bins_var = tk.IntVar()
        equalization_bins_var.set(5)  # Set the initial value
        equalization_bins_entry = tk.Entry(
            watershed_settings_window, textvariable=equalization_bins_var
        )
        equalization_bins_entry.pack()

        # Create intensity threshold input
        intensity_thresh_label = tk.Label(
            watershed_settings_window, text="Intensity Threshold:"
        )
        intensity_thresh_label.pack()
        intensity_thresh_var = tk.StringVar()
        intensity_thresh_var.set(
            ",".join(map(str, INTENSITY_THRESHOLDS[0]))
        )  # Set the initial value
        intensity_thresh_entry = tk.Entry(
            watershed_settings_window, textvariable=intensity_thresh_var
        )
        intensity_thresh_entry.pack()

        # Create size threshold input
        size_thresh_label = tk.Label(watershed_settings_window, text="Size Threshold:")
        size_thresh_label.pack()
        size_thresh_var = tk.StringVar()
        size_thresh_var.set(
            ",".join(map(str, SIZE_THRESHOLDS[2]))
        )  # Set the initial value
        size_thresh_entry = tk.Entry(
            watershed_settings_window, textvariable=size_thresh_var
        )
        size_thresh_entry.pack()

        # Create max aspect ratio input
        max_aspect_ratio_label = tk.Label(
            watershed_settings_window, text="Max Aspect Ratio:"
        )
        max_aspect_ratio_label.pack()
        max_aspect_ratio_var = tk.DoubleVar()
        max_aspect_ratio_var.set(2.5)  # Set the initial value
        max_aspect_ratio_entry = tk.Entry(
            watershed_settings_window, textvariable=max_aspect_ratio_var
        )
        max_aspect_ratio_entry.pack()

        # Create min solidity input
        min_solidity_label = tk.Label(watershed_settings_window, text="Min Solidity:")
        min_solidity_label.pack()
        min_solidity_var = tk.DoubleVar()
        min_solidity_var.set(0.55)  # Set the initial value
        min_solidity_entry = tk.Entry(
            watershed_settings_window, textvariable=min_solidity_var
        )
        min_solidity_entry.pack()

        # Create min area input
        min_area_label = tk.Label(watershed_settings_window, text="Min Area:")
        min_area_label.pack()
        min_area_var = tk.DoubleVar()
        min_area_var.set(200)  # Set the initial value
        min_area_entry = tk.Entry(watershed_settings_window, textvariable=min_area_var)
        min_area_entry.pack()

        # Create save button
        def save_settings():
            stain_vector_index = stain_vector_var.get()
            if stain_vector_index in STAIN_VECTORS:
                self.watershed_settings = {
                    "stain_vector": stain_vector_index,
                    "equalization_bins": equalization_bins_var.get(),
                    "intensity_thresh": tuple(
                        map(float, intensity_thresh_var.get().split(","))
                    ),
                    "size_thresh": tuple(map(float, size_thresh_var.get().split(","))),
                    "max_aspect_ratio": max_aspect_ratio_var.get(),
                    "min_solidity": min_solidity_var.get(),
                    "min_area": min_area_var.get(),
                    # "dist_thresh": dist_thresh_var.get()
                }
            else:
                logging.warning(f"Invalid stain vector index: {stain_vector_index}")

        save_button = tk.Button(
            watershed_settings_window, text="Save", command=save_settings
        )
        save_button.pack(pady=10)

        watershed_settings_window.mainloop()

    def run_sam_with_selected_annotations(self):
        selected_indices = self.annotation_listbox.curselection()
        selected_points = [
            self.points[i] for i in selected_indices if i < len(self.points)
        ]
        selected_boxes = [
            self.boxes[i - len(self.points)]
            for i in selected_indices
            if i >= len(self.points) and i < len(self.points) + len(self.boxes)
        ]

        if not selected_points and not selected_boxes:
            logging.warning(
                "No annotations selected. Please select at least one annotation."
            )
            return

        self.save_selected_annotations(selected_points, selected_boxes)
        path_to_weights = self.sam_weights_path

        self.create_loading_screen("Running SAM.\nThis may take some time...")
        self.update()

        output_dir = sam_main(
            path_to_weights,
            annotations_filename="selected_annotations.npz",
            image_folder=self.image_folder,
            model_size=self.sam_model_size,
        )
        logging.info("Segmentation completed with selected annotations.")

        self.loading_screen.destroy()

        # Save SAM-generated masks with a timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
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

    def save_selected_annotations(self, selected_points, selected_boxes):
        if self.opened_image:
            default_filename = "selected_annotations.npz"
            points_array = np.array(selected_points)
            boxes_array = np.array(selected_boxes)
            np.savez(
                default_filename,
                points=points_array,
                boxes=boxes_array,
                image_path=self.image_path,
            )
            logging.info(
                f"Selected annotations saved successfully as {default_filename}."
            )
        else:
            logging.warning(
                "No image opened. Please open an image before saving annotations."
            )

    def run_watershed(self):
        # Create loading window
        if self.opened_image:
            self.create_loading_screen("Running Watershed.\nThis may take some time...")
            self.update()

            if self.image_path.endswith(".czi"):
                czi = CziFile(self.image_path)
                # image is in ((time, Y, X, channel), metadata) format
                image = czi.read_image()[0][-1, :]
                # normalize to 255 (data appears to be in uint16)
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image = cv2.imread(self.image_path)
            # Get centroids

            stain_vector = STAIN_VECTORS[self.watershed_settings["stain_vector"]]
            centroid_coords, deconv_img, segmented_img, distances = generate_centroid(
                image,
                stain_vector,
                self.watershed_settings["equalization_bins"],
                self.watershed_settings["intensity_thresh"],
                self.watershed_settings["size_thresh"],
                self.watershed_settings["max_aspect_ratio"],
                self.watershed_settings["min_solidity"],
                self.watershed_settings["min_area"],
                # self.watershed_settings["dist_thresh"]
            )

            # Clear existing point annotations
            self.clear_points_and_boxes()

            # Add centroid coordinates as point annotations
            for coord in centroid_coords:
                x, y = coord
                oval_id = self.canvas.create_oval(
                    x + self.drag_coefficient_x - 2,
                    y + self.drag_coefficient_y - 2,
                    x + self.drag_coefficient_x + 2,
                    y + self.drag_coefficient_y + 2,
                    fill="red",
                )
                self.points.append((x, y))
                self.point_ids.append(oval_id)

            self.update_annotation_listbox()

            self.loading_screen.destroy()
        else:
            logging.warning("No image opened. Please open an image first.")

    def show_sam_settings(self):
        sam_settings_window = tk.Toplevel(self)
        sam_settings_window.title("SAM Settings")

        # Create model size input
        model_size_label = tk.Label(sam_settings_window, text="Model Size:")
        model_size_label.pack()
        model_size_var = tk.StringVar()
        model_size_var.set(
            self.sam_model_size
        )  # Set the initial value from the instance variable
        model_size_entry = tk.Entry(sam_settings_window, textvariable=model_size_var)
        model_size_entry.pack()

        # Create path to weights input
        weights_path_label = tk.Label(sam_settings_window, text="Path to Weights:")
        weights_path_label.pack()
        weights_path_var = tk.StringVar()
        weights_path_var.set(
            self.sam_weights_path
        )  # Set the initial value from the instance variable
        weights_path_entry = tk.Entry(
            sam_settings_window, textvariable=weights_path_var
        )
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
        image_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.czi")]
        )
        if image_path:
            self.image_path = image_path
            self.image_name = os.path.splitext(os.path.basename(image_path))[0]
            self.create_unique_image_folder()
            self.copy_image_to_folder()

            self.pixel_to_unit_scale = 1
            unit_txt = "unit measurement"

            # Special processing for .czi files
            if image_path.endswith(".czi"):
                czi = CziFile(self.image_path)
                # image is in ((time, Y, X, channel), metadata) format
                image = czi.read_image()[0][-1, :]

                # normalize to 255 (data appears to be in uint16)
                image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

                # Set a default resolution to x=y=1 if not available

                metadata = czi.meta
                tree = ET.ElementTree(metadata)
                node_dist_x = tree.find(".//Distance[@Id='X']")

                # x=y assumed to be same for this impl
                # node_dist_y = tree.find(".//Distance[@Id='Y']")

                # resolution conversion
                if node_dist_x:
                    scale = node_dist_x.find("Value")
                    unit = node_dist_x.find("DefaultUnitFormat")

                    # This value should be equivalent to what is observed when image loaded into FIJI
                    self.pixel_to_unit_scale = (
                        (float(scale.text) * 1e6)
                        if scale is not None
                        else self.pixel_to_unit_scale
                    )

                    unit_txt = unit.text if unit is not None else unit_txt

                logging.info(
                    f"CZI loaded, pixel-to-measurement scale is set to {self.pixel_to_unit_scale:.4f} {unit_txt} / pixel."
                )

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
            else:
                image = Image.open(
                    os.path.join(self.cache_folder, os.path.basename(image_path))
                )

                # This is not the right "resolution".
                # self.pixel_to_unit_scale, _ = image.info.get("resolution", (1, 1))
                # self.pixel_to_unit_scale = 1 / float(self.pixel_to_unit_scale)

                logging.info(
                    (
                        "Non-CZI file selected,"
                        f"pixel-to-measurement scale is set to {self.pixel_to_unit_scale:.4f} / {unit_txt}."
                    )
                )

            self.opened_image = image
            photo = ImageTk.PhotoImage(image)
            self.canvas.image = photo
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)
            self.clear_annotations()
            self.drag_coefficient_x = 0
            self.drag_coefficient_y = 0
            self.minimap_drag_coefficient_x = 0
            self.minimap_drag_coefficient_y = 0

            logging.info(f"Image opened: {image_path}")
        else:
            self.opened_image = None
            self.canvas.delete("all")
            logging.warning("No image selected.")

    def create_unique_image_folder(self):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.image_folder = f"image_masks/{self.image_name}_{timestamp}"
        self.cache_folder = f"cache/{self.image_name}_{timestamp}"
        os.makedirs(self.image_folder, exist_ok=True)
        os.makedirs(self.cache_folder, exist_ok=True)

    def copy_image_to_folder(self):
        shutil.copy2(self.image_path, self.cache_folder)

    def load_masks(self):
        self.masks = []
        self.mask_files = []
        all_files = [f for f in os.listdir(self.image_folder) if f.endswith(".png")]
        mask_files = [
            f
            for f in all_files
            if not f.startswith(self.image_name)
            and not f.startswith("predicted_")
            and not f.startswith("mask_")
        ]
        mask_files = sorted(
            mask_files,
            key=lambda x: os.path.getmtime(os.path.join(self.image_folder, x)),
        )
        for mask_file in mask_files:
            mask_path = os.path.join(self.image_folder, mask_file)
            mask = cv2.imread(mask_path, 0)
            self.masks.append(mask)
            self.mask_files.append(mask_file)
        self.display_masks()
        self.update_annotation_listbox()

    def create_loading_screen(self, text):
        self.loading_screen = tk.Toplevel(self)
        self.loading_screen.title("Loading")
        self.loading_screen.geometry("200x100")
        self.loading_screen.resizable(False, False)

        label = tk.Label(self.loading_screen, text=text)
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
            mask_rgba[..., 3] = (mask > 0).astype(
                np.uint8
            ) * 128  # Set alpha channel based on mask
            mask_image = Image.fromarray(mask_rgba, mode="RGBA")
            mask_photo = ImageTk.PhotoImage(mask_image)
            self.canvas.mask_images.append(
                mask_photo
            )  # Keep a reference to the mask photo
            self.canvas.create_image(
                self.drag_coefficient_x,
                self.drag_coefficient_y,
                anchor=tk.NW,
                image=mask_photo,
                tags=f"mask_{i}",
            )

    def toggle_manual_mask_mode(self):
        if self.opened_image:
            self.manual_mask_mode = not self.manual_mask_mode
            self.box_select_mode = False
            self.point_select_mode = False
            self.drag_mode = False
            if self.manual_mask_mode:
                self.manual_mask_button.configure(bg="lightblue")
                self.box_select_button.configure(bg="lightgray")
                self.point_select_button.configure(bg="lightgray")
                self.drag_button.configure(bg="lightgray")
            else:
                self.manual_mask_button.configure(bg="lightgray")
        else:
            logging.warning("No image opened. Please open an image first.")

    def display_masks(self):
        for i, mask in enumerate(self.masks):
            binary_mask = mask > 0
            mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
            mask_rgba[..., :3] = [30, 144, 255]  # Blue color
            mask_rgba[..., 3] = (
                binary_mask.astype(np.uint8) * 128
            )  # Set alpha channel based on mask

            mask_image = Image.fromarray(mask_rgba, mode="RGBA")
            mask_photo = ImageTk.PhotoImage(mask_image)
            self.canvas.mask_images.append(
                mask_photo
            )  # Keep a reference to the mask photo
            self.canvas.create_image(
                self.drag_coefficient_x,
                self.drag_coefficient_y,
                anchor=tk.NW,
                image=mask_photo,
                tags=f"mask_{i}",
            )

    def toggle_box_select_mode(self):
        if self.opened_image:
            self.box_select_mode = not self.box_select_mode
            self.point_select_mode = False
            self.drag_mode = False
            self.manual_mask_mode = False
            if self.box_select_mode:
                self.box_select_button.configure(bg="lightblue")
                self.point_select_button.configure(bg="lightgray")
                self.drag_button.configure(bg="lightgray")
                self.manual_mask_button.configure(bg="lightgray")
            else:
                self.box_select_button.configure(bg="lightgray")
        else:
            logging.warning("No image opened. Please open an image first.")

    def toggle_point_select_mode(self):
        if self.opened_image:
            self.point_select_mode = not self.point_select_mode
            self.box_select_mode = False
            self.drag_mode = False
            self.manual_mask_mode = False
            if self.point_select_mode:
                self.point_select_button.configure(bg="lightblue")
                self.box_select_button.configure(bg="lightgray")
                self.drag_button.configure(bg="lightgray")
                self.manual_mask_button.configure(bg="lightgray")
            else:
                self.point_select_button.configure(bg="lightgray")
        else:
            logging.warning("No image opened. Please open an image first.")

    def on_canvas_click(self, event):
        if self.opened_image:
            if self.drag_mode:
                self.drag_start_x = event.x
                self.drag_start_y = event.y
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
                oval_id = self.canvas.create_oval(
                    x - 2, y - 2, x + 2, y + 2, fill="red"
                )
                self.points.append(
                    (x - self.drag_coefficient_x, y - self.drag_coefficient_y)
                )
                self.point_ids.append(oval_id)
                self.update_annotation_listbox()
            else:
                self.check_annotation_click(event.x, event.y)

    def on_canvas_drag(self, event):
        if self.opened_image:
            if self.drag_mode:
                delta_x = event.x - self.drag_start_x
                delta_y = event.y - self.drag_start_y
                self.canvas.move("all", delta_x, delta_y)
                self.canvas.move("mask_highlight", delta_x, delta_y)
                self.drag_start_x = event.x
                self.drag_start_y = event.y
                self.drag_coefficient_x += delta_x
                self.drag_coefficient_y += delta_y

                # Update minimap drag coefficients
                self.minimap_drag_coefficient_x -= delta_x
                self.minimap_drag_coefficient_y -= delta_y

                self.update_minimap_rect(event)

        if self.box_select_mode:
            if self.rect_id:
                self.canvas.delete(self.rect_id)
            self.rect_id = self.canvas.create_rectangle(
                self.start_x, self.start_y, event.x, event.y, outline="red"
            )
        elif self.manual_mask_mode:
            self.canvas.delete("manual_mask")
            self.manual_mask_path.append((event.x, event.y))
            self.canvas.create_line(
                self.manual_mask_path,
                fill="red",
                tags="manual_mask",
                width=self.brush_size,
            )

    def on_canvas_release(self, event):
        if self.opened_image:
            if self.drag_mode:
                delta_x = event.x - self.drag_start_x
                delta_y = event.y - self.drag_start_y
                self.canvas.move("all", delta_x, delta_y)
            if self.manual_mask_mode:
                self.manual_mask_path.append((event.x, event.y))
                self.create_manual_mask()
            elif self.box_select_mode:
                x1 = min(self.start_x, event.x)
                y1 = min(self.start_y, event.y)
                x2 = max(self.start_x, event.x)
                y2 = max(self.start_y, event.y)
                self.boxes.append(
                    (
                        x1 - self.drag_coefficient_x,
                        y1 - self.drag_coefficient_y,
                        x2 - self.drag_coefficient_x,
                        y2 - self.drag_coefficient_y,
                    )
                )
                self.box_ids.append(self.rect_id)
                self.rect_id = None  # Reset rect_id after appending it to box_ids
                self.update_annotation_listbox()

    def create_manual_mask(self):
        mask = np.zeros(
            (self.opened_image.height, self.opened_image.width), dtype=np.uint8
        )
        manual_mask_polygon = np.array(
            [
                (x - self.drag_coefficient_x, y - self.drag_coefficient_y)
                for x, y in self.manual_mask_path
            ],
            dtype=np.int32,
        )
        cv2.polylines(
            mask, [manual_mask_polygon], False, 255, thickness=self.brush_size
        )

        # Save the manual mask to the image folder with a timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
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
        for i, point in enumerate(self.points):
            self.annotation_listbox.insert(
                tk.END, f"Point {i + 1}: ({point[0]}, {point[1]})"
            )
        for i, box in enumerate(self.boxes):
            self.annotation_listbox.insert(
                tk.END, f"Box {i + 1}: ({box[0]}, {box[1]}, {box[2]}, {box[3]})"
            )
        for i, mask_file in enumerate(self.mask_files):
            self.annotation_listbox.insert(tk.END, f"Mask {i + 1}: {mask_file}")

    def on_annotation_select(self, event):
        selection = self.annotation_listbox.curselection()
        self.highlight_annotations(selection)
        if self.multi_select_mode:
            # Highlight multiple selected masks
            mask_indices = [index - len(self.points) - len(self.boxes) for index in selection if
                            index >= len(self.points) + len(self.boxes)]
            self.highlight_masks(mask_indices)
        else:
            # Highlight the selected mask
            if len(selection) == 1:
                index = selection[0]
                if index < len(self.points):
                    self.highlight_point(index)
                elif index < len(self.points) + len(self.boxes):
                    self.highlight_box(index - len(self.points))
                else:
                    mask_index = index - len(self.points) - len(self.boxes)
                    self.highlight_mask(mask_index)
            else:
                self.highlight_mask(-1)  # Clear highlight if multiple items are selected

    def highlight_point(self, point_index):
        self.canvas.delete("highlight")  # Remove any existing highlight
        if point_index >= 0 and point_index < len(self.points):
            point = self.points[point_index]
            self.canvas.create_oval(
                point[0] + self.drag_coefficient_x - 4,
                point[1] + self.drag_coefficient_y - 4,
                point[0] + self.drag_coefficient_x + 4,
                point[1] + self.drag_coefficient_y + 4,
                outline="yellow",
                tags="highlight",
            )

    def highlight_box(self, box_index):
        self.canvas.delete("highlight")  # Remove any existing highlight
        if box_index >= 0 and box_index < len(self.boxes):
            box = self.boxes[box_index]
            self.canvas.create_rectangle(
                box[0] + self.drag_coefficient_x,
                box[1] + self.drag_coefficient_y,
                box[2] + self.drag_coefficient_x,
                box[3] + self.drag_coefficient_y,
                outline="yellow",
                tags="highlight",
            )

    def highlight_mask(self, mask_index):
        self.canvas.delete("mask_highlight")
        if mask_index >= 0 and mask_index < len(self.masks):
            mask = self.masks[mask_index]
            mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
            mask_rgba[..., :3] = [255, 255, 0]  # Yellow color for highlight
            mask_rgba[..., 3] = (mask > 0).astype(np.uint8) * 128  # Set alpha channel based on mask
            mask_image = Image.fromarray(mask_rgba, mode="RGBA")
            mask_photo = ImageTk.PhotoImage(mask_image)
            self.canvas.mask_highlight_image = mask_photo  # Keep a reference to the mask photo
            self.canvas.create_image(
                self.drag_coefficient_x,
                self.drag_coefficient_y,
                anchor=tk.NW,
                image=mask_photo,
                tags="mask_highlight",
            )

    def highlight_masks(self, mask_indices):
        self.canvas.delete("mask_highlight")
        self.canvas.mask_highlight_images.clear()  # Clear the list before appending new mask photos
        for mask_index in mask_indices:
            if mask_index >= 0 and mask_index < len(self.masks):
                mask = self.masks[mask_index]
                mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
                mask_rgba[..., :3] = [255, 255, 0]  # Yellow color for highlight
                mask_rgba[..., 3] = (mask > 0).astype(np.uint8) * 128  # Set alpha channel based on mask
                mask_image = Image.fromarray(mask_rgba, mode="RGBA")
                mask_photo = ImageTk.PhotoImage(mask_image)
                self.canvas.mask_highlight_images.append(mask_photo)  # Keep a reference to the mask photo
                self.canvas.create_image(
                    self.drag_coefficient_x,
                    self.drag_coefficient_y,
                    anchor=tk.NW,
                    image=mask_photo,
                    tags="mask_highlight",
                )

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
                self.canvas.create_oval(
                    point[0] + self.drag_coefficient_x - 4,
                    point[1] + self.drag_coefficient_y - 4,
                    point[0] + self.drag_coefficient_x + 4,
                    point[1] + self.drag_coefficient_y + 4,
                    outline="yellow",
                    tags="highlight",
                )
            else:
                box_index = index - len(self.points)
                if box_index < len(self.boxes):
                    box = self.boxes[box_index]
                    self.canvas.create_rectangle(
                        box[0] + self.drag_coefficient_x,
                        box[1] + self.drag_coefficient_y,
                        box[2] + self.drag_coefficient_x,
                        box[3] + self.drag_coefficient_y,
                        outline="yellow",
                        tags="highlight",
                    )

    def check_annotation_click(self, x, y):
        # Check if a point is clicked
        for i, point_id in enumerate(self.point_ids):
            coords = self.canvas.coords(point_id)
            if coords[0] <= x <= coords[2] and coords[1] <= y <= coords[3]:
                self.annotation_listbox.selection_clear(0, tk.END)
                self.annotation_listbox.selection_set(i)
                self.highlight_point(i)
                return

        # Check if a box is clicked
        for i, box_id in enumerate(self.box_ids):
            coords = self.canvas.coords(box_id)
            if coords[0] <= x <= coords[2] and coords[1] <= y <= coords[3]:
                self.annotation_listbox.selection_clear(0, tk.END)
                self.annotation_listbox.selection_set(len(self.points) + i)
                self.highlight_box(i)
                return

        selected_mask_indices = []
        for i, mask in enumerate(self.masks):
            if mask[y - self.drag_coefficient_y, x - self.drag_coefficient_x] > 0:
                if self.multi_select_mode:
                    index = len(self.points) + len(self.boxes) + i
                    if index in self.annotation_listbox.curselection():
                        self.annotation_listbox.selection_clear(index)
                    else:
                        self.annotation_listbox.selection_set(index)
                    selected_mask_indices = [idx - len(self.points) - len(self.boxes) for idx in
                                             self.annotation_listbox.curselection() if
                                             idx >= len(self.points) + len(self.boxes)]
                else:
                    self.annotation_listbox.selection_clear(0, tk.END)
                    self.annotation_listbox.selection_set(len(self.points) + len(self.boxes) + i)
                    selected_mask_indices = [i]
                break

        if selected_mask_indices:
            self.highlight_masks(selected_mask_indices)
        else:
            self.annotation_listbox.selection_clear(0, tk.END)
            self.highlight_mask(-1)  # Clear highlight if no annotation is clicked

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
            default_filename = "current_annotations.npz"
            points_array = np.array(self.points)
            boxes_array = np.array(self.boxes)
            np.savez(
                default_filename,
                points=points_array,
                boxes=boxes_array,
                image_path=self.image_path,
            )
            logging.info(f"Annotations saved successfully as {default_filename}.")
        else:
            logging.warning(
                "No image opened. Please open an image before saving annotations."
            )

    def clear_annotations(self):
        self.points = []
        self.boxes = []
        self.update_annotation_listbox()
        self.canvas.delete("annotation")
        if self.opened_image is not None:
            photo = ImageTk.PhotoImage(self.opened_image)
            self.canvas.image = photo
            self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    def run_sam_with_current_annotation(self):
        if self.sam is None:
            self.sam = SAModel()
            model_type = (
                SAModelType.SAM_VIT_L
                if self.sam_model_size == "L"
                else (
                    SAModelType.SAM_VIT_B
                    if self.sam_model_size == "B"
                    else SAModelType.SAM_VIT_H
                )
            )
            logging.info(f"Attempting to load model_type: {model_type}")
            self.sam.load_weights(
                model_type=model_type, path_to_weights=self.sam_weights_path
            )

        self.save_current_annotations()
        path_to_weights = self.sam_weights_path

        self.create_loading_screen("Running SAM.\nThis may take some time...")
        self.update()

        output_dir = sam_main(
            path_to_weights,
            annotations_filename="current_annotations.npz",
            image_folder=self.image_folder,
            model_size=self.sam_model_size,
            sam=self.sam,
        )
        logging.info("Segmentation completed with all annotations.")

        self.loading_screen.destroy()

        # Save SAM-generated masks with a timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
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

    def run_sam_with_selected_annotations(self):
        if self.sam is None:
            self.sam = SAModel()
            model_type = (
                SAModelType.SAM_VIT_L
                if self.sam_model_size == "L"
                else (
                    SAModelType.SAM_VIT_B
                    if self.sam_model_size == "B"
                    else SAModelType.SAM_VIT_H
                )
            )
            logging.info(f"Attempting to load model_type: {model_type}")
            self.sam.load_weights(
                model_type=model_type, path_to_weights=self.sam_weights_path
            )

        selected_indices = self.annotation_listbox.curselection()
        selected_points = [
            self.points[i] for i in selected_indices if i < len(self.points)
        ]
        selected_boxes = [
            self.boxes[i - len(self.points)]
            for i in selected_indices
            if i >= len(self.points) and i < len(self.points) + len(self.boxes)
        ]

        if not selected_points and not selected_boxes:
            logging.warning(
                "No annotations selected. Please select at least one annotation."
            )
            return

        self.save_selected_annotations(selected_points, selected_boxes)
        path_to_weights = self.sam_weights_path

        self.create_loading_screen("Running SAM.\nThis may take some time...")
        self.update()

        output_dir = sam_main(
            path_to_weights,
            annotations_filename="selected_annotations.npz",
            image_folder=self.image_folder,
            model_size=self.sam_model_size,
            sam=self.sam,
        )
        logging.info("Segmentation completed with selected annotations.")

        self.loading_screen.destroy()

        # Save SAM-generated masks with a timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
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
        image_path = os.path.join(masks_dir, "predicted_image.png")
        image, masks, mask_files = self.load_image_with_masks(image_path, masks_dir)

        # Save the predicted_image.png to the cache folder
        shutil.copy2(image_path, os.path.join(self.cache_folder, "predicted_image.png"))

        self.opened_image = Image.fromarray(image)
        photo = ImageTk.PhotoImage(self.opened_image)
        self.canvas.delete("all")
        self.canvas.image = photo
        self.canvas.create_image(0, 0, anchor=tk.NW, image=photo)

        for i, mask in enumerate(masks):
            mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
            mask_rgba[..., :3] = [30, 144, 255]  # Blue color
            mask_rgba[..., 3] = (mask > 0).astype(
                np.uint8
            ) * 128  # Set alpha channel based on mask

            mask_image = Image.fromarray(mask_rgba, mode="RGBA")
            mask_photo = ImageTk.PhotoImage(mask_image)
            self.canvas.mask_images.append(
                mask_photo
            )  # Keep a reference to the mask photo
            self.canvas.create_image(
                0, 0, anchor=tk.NW, image=mask_photo, tags=f"mask_{i}"
            )

        self.masks = masks
        self.mask_files = mask_files
        self.masks_dir = masks_dir
        self.update_annotation_listbox()

    def load_image_with_masks(self, image_path, masks_dir):
        image = cv2.imread(
            os.path.join(self.cache_folder, os.path.basename(image_path))
        )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_files = [
            mask for mask in os.listdir(masks_dir) if mask.startswith("mask_")
        ]
        masks = [
            cv2.imread(os.path.join(masks_dir, mask_file), 0)
            for mask_file in mask_files
        ]
        return image, masks, mask_files

    def run_analysis(self):
        if not self.masks:
            logging.warning("No masks found. Please run SAM first.")
            return

        binary_masks = [mask > 0 for mask in self.masks]

        results = []
        for binary_mask in binary_masks:
            prop_df = get_prop(binary_mask)
            if not prop_df.empty:
                result = analyze_properties(prop_df, self.pixel_to_unit_scale)
                results.append(result)

        results = pd.concat(results, axis=1).T.reset_index(drop=True)
        # Use perimeter to prevent scale issues
        outlier_bools = detect_outliers(results[_MEASURED_PROPS[1]])

        combined_results = pd.concat([results, outlier_bools], axis=1)

        csv_output_folder = filedialog.asksaveasfilename(
            initialdir=".",
            title="Save Location",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )

        if csv_output_folder:
            combined_results.to_csv(csv_output_folder, index=False)

            logging.info(f"Analysis results saved to {csv_output_folder}")
            # print("Analysis Results:")
            # for i, result in enumerate(results):
            #     print(f"Mask {i + 1}:")
            #     print(result)


if __name__ == "__main__":
    app = ImageViewer()
    app.mainloop()
