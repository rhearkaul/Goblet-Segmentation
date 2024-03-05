import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
import cv2
import numpy as np

def load_image_with_masks(image_path, masks_dir):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask_files = [mask for mask in os.listdir(masks_dir) if mask.startswith('mask_')]
    masks = [cv2.imread(os.path.join(masks_dir, mask_file), 0) for mask_file in mask_files]
    return image, masks, mask_files

def update_figure(canvas, figure, image, masks):
    figure.clear()
    ax = figure.add_subplot(111)
    ax.imshow(image)
    for mask in masks:
        ax.imshow(mask, alpha=0.5, cmap='gray')
    ax.axis('off')
    canvas.draw()

def on_delete_mask(masks, mask_listbox, canvas, figure, image, masks_dir):
    selected_indices = mask_listbox.curselection()
    if not selected_indices:
        messagebox.showinfo("Info", "Please select a mask to delete.")
        return

    for index in reversed(selected_indices):
        mask_filename = mask_listbox.get(index)
        os.remove(os.path.join(masks_dir, mask_filename))
        del masks[index]
        mask_listbox.delete(index)

    update_figure(canvas, figure, image, masks)

def main():
    root = tk.Tk()
    root.title("Image and Mask Viewer")

    figure = plt.Figure()
    canvas = FigureCanvasTkAgg(figure, root)
    canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    image_path = filedialog.askopenfilename(title="Select the Predicted Image", filetypes=[("PNG files", "*.png")])
    masks_dir = filedialog.askdirectory(title="Select the Masks Directory")
    image, masks, mask_files = load_image_with_masks(image_path, masks_dir)
    update_figure(canvas, figure, image, masks)

    mask_list_frame = tk.Frame(root)
    mask_list_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    mask_listbox = tk.Listbox(mask_list_frame, selectmode=tk.MULTIPLE, width=50, height=20)
    mask_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar = tk.Scrollbar(mask_list_frame, orient="vertical")
    scrollbar.config(command=mask_listbox.yview)
    scrollbar.pack(side=tk.RIGHT, fill="y")
    mask_listbox.config(yscrollcommand=scrollbar.set)

    for mask_file in mask_files:
        mask_listbox.insert(tk.END, mask_file)

    delete_button = tk.Button(root, text="Delete Selected Mask(s)", command=lambda: on_delete_mask(masks, mask_listbox, canvas, figure, image, masks_dir))
    delete_button.pack(side=tk.TOP)

    root.mainloop()

if __name__ == '__main__':
    main()

