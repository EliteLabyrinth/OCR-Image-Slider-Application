import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os


class ImageSlider:
    def __init__(self, root, image_paths):
        self.root = root
        self.root.title("Image Slider")
        self.image_paths = image_paths
        self.current_image_index = 0

        # Load the first image
        self.load_image()

        # Create a label to display the image
        self.image_label = ttk.Label(root, image=self.image_tk)
        self.image_label.pack(expand=True)

        self.image_number_label = ttk.Label(root, text="")
        self.image_number_label.pack(side=tk.BOTTOM, pady=10)
        self.update_image_number_label()

        # Create a frame for buttons
        button_frame = ttk.Frame(root)
        button_frame.pack(side=tk.BOTTOM, pady=10)

        # Create Previous Button
        self.prev_button = ttk.Button(
            button_frame, text="Previous", command=self.show_previous_image
        )
        self.prev_button.pack(side=tk.LEFT, padx=10)

        # Create Next Button
        self.next_button = ttk.Button(
            button_frame, text="Next", command=self.show_next_image
        )
        self.next_button.pack(side=tk.RIGHT, padx=10)

        # Create Grayscale Button
        self.grayscale_button = ttk.Button(
            root, text="Grayscale", command=self.convert_to_grayscale
        )
        self.grayscale_button.pack(side=tk.BOTTOM, pady=10)

        scrollbar = tk.Scrollbar(root, orient="vertical")
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def load_image(self):
        # Open the image
        self.image = Image.open(self.image_paths[self.current_image_index])

        # Resize the image to fit the viewport while maintaining aspect ratio
        self.image = self.resize_image(self.image)

        # Convert to PhotoImage for Tkinter
        self.image_tk = ImageTk.PhotoImage(self.image)

    def resize_image(self, image):
        # Get the size of the root window
        window_width = self.root.winfo_width()
        window_height = self.root.winfo_height()

        if (
            window_width == 1 and window_height == 1
        ):  # Initial size before window is shown
            window_width = 800
            window_height = 600

        # Calculate the aspect ratio
        aspect_ratio = image.width / image.height

        # Determine the new size maintaining aspect ratio
        if window_width / aspect_ratio < window_height:
            new_width = window_width
            new_height = int(window_width / aspect_ratio)
        else:
            new_width = int(window_height * aspect_ratio)
            new_height = window_height

        return image.resize((new_width, new_height - 200))

    def show_previous_image(self):
        self.current_image_index = (self.current_image_index - 1) % len(
            self.image_paths
        )
        self.load_image()
        self.image_label.config(image=self.image_tk)
        self.update_image_number_label()

    def show_next_image(self):
        self.current_image_index = (self.current_image_index + 1) % len(
            self.image_paths
        )
        self.load_image()
        self.image_label.config(image=self.image_tk)
        self.update_image_number_label()

    def convert_to_grayscale(self):
        grayscale_image = self.image.convert("L")
        self.image_tk = ImageTk.PhotoImage(grayscale_image)
        self.image_label.config(image=self.image_tk)

    def update_image_number_label(self):
        total_images = len(self.image_paths)
        current_image_number = self.current_image_index + 1
        self.image_number_label.config(
            text=f"Image {current_image_number} / {total_images}"
        )


if __name__ == "__main__":
    # Example list of image paths
    image_dir = (
        "./sampleImages/manhua"  # Set this to the directory containing your images
    )
    image_paths = [
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.endswith(("png", "jpg", "jpeg", "bmp", "gif"))
    ]

    # Create the main window
    root = tk.Tk()
    root.geometry("800x600")  # Set initial size of the window

    # Initialize the ImageSlider
    slider = ImageSlider(root, image_paths)

    # Start the GUI event loop
    root.mainloop()
