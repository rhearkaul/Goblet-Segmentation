import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image

class ImageViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Viewer")

        # Get the current screen resolution
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()

        # Set the window size to 75% of the screen resolution
        window_width = int(screen_width * 0.75)
        window_height = int(screen_height * 0.75)

        self.geometry(f"{window_width}x{window_height}")
        self.resizable(False, False)

        self.create_widgets(window_width, window_height)

    def create_widgets(self, window_width, window_height):
        # Create toolbar frame
        toolbar_frame = tk.Frame(self, bg="gray")
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)

        input_button = tk.Button(toolbar_frame, text="Step 1: Input", command=self.open_image)
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

        box_select_button = tk.Button(select_toolbar_frame, text="Box Select")
        box_select_button.pack(side=tk.LEFT, padx=5, pady=5)

        point_select_button = tk.Button(select_toolbar_frame, text="Point Select")
        point_select_button.pack(side=tk.LEFT, padx=5, pady=5)

        function_window_width = int(window_width * 0.2)
        function_window_height = window_height - 80
        image_display_width = window_width - function_window_width
        image_display_height = window_height - 80

        function_window_frame = tk.Frame(self, bg="white", width=function_window_width, height=function_window_height)
        function_window_frame.pack(side=tk.LEFT, fill=tk.BOTH)
        function_window_frame.pack_propagate(False)

        self.image_viewer_frame = tk.Frame(self, bg="white", width=image_display_width, height=image_display_height)
        self.image_viewer_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
        self.image_viewer_frame.pack_propagate(False)

        self.image_label = tk.Label(self.image_viewer_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True)

    def open_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            image = Image.open(file_path)
            image.thumbnail((600, 400))
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo)
            self.image_label.image = photo

if __name__ == "__main__":
    app = ImageViewer()
    app.mainloop()


