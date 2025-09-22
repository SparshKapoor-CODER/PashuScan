import tkinter as tk
from tkinter import messagebox, simpledialog
from PIL import Image, ImageTk
import numpy as np
import cv2

class ManualCalibrationDialog:
    """
    Simple helper to let the user click two points on an image displayed in a Tk window,
    then enter the actual distance (cm). Returns calibration_factor (cm per px) or None.
    """

    def __init__(self, parent, pil_image):
        """
        parent: tk root or Toplevel
        pil_image: PIL.Image in RGB mode (original size)
        """
        self.parent = parent
        self.pil_image = pil_image
        self.clicks = []
        self.result = None

        self.win = tk.Toplevel(parent)
        self.win.title("Manual Calibration - Click two points")
        # show a resized preview for easier clicking if image is large
        preview = pil_image.copy()
        preview.thumbnail((800, 800))
        self.display_size = preview.size  # (w,h)
        self.photo = ImageTk.PhotoImage(preview)
        self.canvas = tk.Canvas(self.win, width=self.display_size[0], height=self.display_size[1], bg="black")
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
        self.canvas.bind("<Button-1>", self.on_click)

        self.win.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_click(self, event):
        x, y = event.x, event.y
        self.clicks.append((x, y))
        # draw small marker
        self.canvas.create_oval(x-4, y-4, x+4, y+4, outline="red", width=2)
        if len(self.clicks) == 2:
            # map clicks on preview back to original image coords
            ox, oy = self.pil_image.size
            dx, dy = self.display_size
            sx = ox / dx
            sy = oy / dy
            (x1, y1), (x2, y2) = self.clicks
            px_dist = np.hypot((x1 - x2) * sx, (y1 - y2) * sy)
            # ask for actual distance
            val = simpledialog.askfloat("Enter real distance", "Enter actual distance between the two points (in cm):", parent=self.win)
            if val is None:
                messagebox.showinfo("Calibration", "Calibration cancelled.")
                self.result = None
            else:
                if px_dist <= 0:
                    messagebox.showerror("Calibration", "Pixel distance is zero; try again.")
                    self.result = None
                else:
                    cm_per_px = val / px_dist
                    self.result = cm_per_px
                    messagebox.showinfo("Calibration", f"Calibration complete: {cm_per_px:.6f} cm/px")
            self.win.destroy()

    def on_close(self):
        self.result = None
        self.win.destroy()

    def show(self):
        self.parent.wait_window(self.win)
        return self.result
