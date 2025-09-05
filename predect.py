# predict.py
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageTk
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import numpy as np
import cv2

# Import measurement and data manager modules
from measure import CattleMeasurements
from data_manager import CattleDataManager
from calibration import ManualCalibrationDialog

# ====================== CONFIG ======================
MODEL_PATH = "dog_breed_model.pth"
LABELS_FILE = "labels.xlsx"
IMG_SIZE = 128
TOPK = 3
RETRAIN_QUEUE = "retrain_queue.csv"
# ====================================================

# ------------------- CNN Model ----------------------
class DogBreedCNN(nn.Module):
    def __init__(self, num_classes):
        super(DogBreedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ----------------- Prediction Helpers ----------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_breed_topk(image_path, model, idx_to_breed, device, topk=3):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probs = F.softmax(output[0], dim=0)
        top_probs, top_idxs = torch.topk(probs, k=min(topk, probs.shape[0]))
        results = [(idx_to_breed[int(i.item())], float(p.item())) for i, p in zip(top_idxs, top_probs)]
    return results

# ----------------- GUI Application -------------------
class BreedPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cattle Breed & Measurement Tool")
        self.root.geometry("900x720")

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Measurement and data manager
        self.measure = CattleMeasurements(calibration_factor=None)
        self.data_manager = CattleDataManager()

        # Load model & mapping (if missing, user can hot-swap)
        self.model = None
        self.idx_to_breed = {}
        try:
            self.model, self.idx_to_breed = self.load_model_and_mapping(MODEL_PATH, LABELS_FILE)
            self.model_loaded_from = MODEL_PATH
        except Exception as e:
            messagebox.showwarning("Model load", f"Could not load default model/labels:\n{e}\nUse Load Model/Labels to choose files.")
            self.model_loaded_from = None

        # UI elements
        self.current_image_path = None
        self.current_pil_image = None    # full-size PIL image
        self.display_size = None         # displayed thumbnail size (w,h)
        self.create_widgets()
        self.update_status()

        # manual calibration state
        self.manual_cal_mode = False
        self.manual_clicks = []

    def load_model_and_mapping(self, model_path, labels_file):
        # Read labels (supports xlsx or csv)
        if not os.path.exists(labels_file):
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        if labels_file.lower().endswith('.csv'):
            df = pd.read_csv(labels_file)
        else:
            df = pd.read_excel(labels_file)

        if 'breed' not in df.columns:
            raise ValueError("Labels file must contain a 'breed' column")

        all_breeds = sorted(df["breed"].unique())
        idx_to_breed = {i: breed for i, breed in enumerate(all_breeds)}
        num_classes = len(all_breeds)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = DogBreedCNN(num_classes)
        state = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model, idx_to_breed

    def create_widgets(self):
        # Title
        title_label = tk.Label(self.root, text="Cattle Breed & Measurement Tool", font=("Arial", 18, "bold"))
        title_label.pack(pady=8)

        main_frame = tk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10)

        # Left: image display
        left = tk.Frame(main_frame)
        left.grid(row=0, column=0, sticky="n", padx=8, pady=4)

        self.image_label = tk.Label(left, text="No image selected", relief="solid", width=48, height=20, bg="white")
        self.image_label.pack()
        self.image_label.bind("<Button-1>", self.on_image_click)  # used for manual calibration clicks

        # Right: controls & results
        right = tk.Frame(main_frame)
        right.grid(row=0, column=1, sticky="nw", padx=8, pady=4)

        # Buttons
        select_btn = tk.Button(right, text="Select Image", command=self.select_image, width=24)
        select_btn.grid(row=0, column=0, pady=4)

        load_model_btn = tk.Button(right, text="Load Model/Labels", command=self.load_model_dialog, width=24)
        load_model_btn.grid(row=1, column=0, pady=4)

        batch_btn = tk.Button(right, text="Batch Predict Folder", command=self.batch_predict_folder, width=24)
        batch_btn.grid(row=2, column=0, pady=4)

        aruco_btn = tk.Button(right, text="Auto Calibrate (ArUco)", command=self.auto_calibrate_aruco, width=24)
        aruco_btn.grid(row=3, column=0, pady=4)

        manual_cal_btn = tk.Button(right, text="Manual Calibrate (2-click)", command=self.start_manual_calibration, width=24)
        manual_cal_btn.grid(row=4, column=0, pady=4)

        measure_btn = tk.Button(right, text="Extract Measurements", command=self.extract_measurements_from_current, width=24)
        measure_btn.grid(row=5, column=0, pady=8)

        save_eval_btn = tk.Button(right, text="Save Evaluation", command=self.save_current_evaluation, width=24)
        save_eval_btn.grid(row=6, column=0, pady=4)

        gradcam_btn = tk.Button(right, text="Show Grad-CAM", command=self.show_gradcam, width=24)
        gradcam_btn.grid(row=7, column=0, pady=4)

        flag_btn = tk.Button(right, text="Flag Incorrect", command=self.flag_incorrect, width=24)
        flag_btn.grid(row=8, column=0, pady=4)

        # TopK control
        tk.Label(right, text="Top-K predictions:").grid(row=9, column=0, pady=(12,0))
        self.topk_var = tk.IntVar(value=TOPK)
        topk_spin = tk.Spinbox(right, from_=1, to=10, textvariable=self.topk_var, width=6)
        topk_spin.grid(row=10, column=0)

        # Results frame
        results_frame = tk.LabelFrame(right, text="Results", padx=6, pady=6)
        results_frame.grid(row=11, column=0, pady=10, sticky="n")

        self.result_label = tk.Label(results_frame, text="Prediction: -", font=("Arial", 12))
        self.result_label.pack(anchor="w")

        self.confidence_label = tk.Label(results_frame, text="Top probabilities: -", font=("Arial", 10))
        self.confidence_label.pack(anchor="w")

        self.length_label = tk.Label(results_frame, text="Body length: -", font=("Arial", 10))
        self.length_label.pack(anchor="w")

        self.height_label = tk.Label(results_frame, text="Height: -", font=("Arial", 10))
        self.height_label.pack(anchor="w")

        self.angle_label = tk.Label(results_frame, text="Rump angle: -", font=("Arial", 10))
        self.angle_label.pack(anchor="w")

        self.calib_label = tk.Label(results_frame, text="Calibration (cm/px): -", font=("Arial", 10))
        self.calib_label.pack(anchor="w")

        self.contour_label = tk.Label(results_frame, text="Contour area (px): -", font=("Arial", 10))
        self.contour_label.pack(anchor="w")

        # Status bar
        self.status_label = tk.Label(self.root, text="", bd=1, relief="sunken", anchor="w")
        self.status_label.pack(side="bottom", fill="x")

    def update_status(self):
        model_info = f"Model: {os.path.basename(self.model_loaded_from)}" if self.model_loaded_from else "Model: (none)"
        classes_info = f" | Classes: {len(self.idx_to_breed)}" if self.idx_to_breed else ""
        calib_info = f" | calib: {self.measure.calibration_factor:.6f} cm/px" if self.measure.calibration_factor else ""
        self.status_label.config(text=model_info + classes_info + calib_info)

    def select_image(self):
        file_path = filedialog.askopenfilename(title="Select an image",
                                               filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return
        try:
            pil = Image.open(file_path).convert("RGB")
            self.current_pil_image = pil.copy()
            display = pil.copy()
            display.thumbnail((560, 560))
            self.display_size = display.size
            photo = ImageTk.PhotoImage(display)
            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # keep reference
            self.current_image_path = file_path

            # Predict if model present
            if self.model is None or not self.idx_to_breed:
                self.result_label.config(text="Prediction: (no model loaded)")
                self.confidence_label.config(text="Top probabilities: -")
            else:
                topk = int(self.topk_var.get())
                results = predict_breed_topk(file_path, self.model, self.idx_to_breed, self.device, topk=topk)
                pred_breed, pred_conf = results[0]
                self.result_label.config(text=f"Prediction: {pred_breed}")
                conf_text = "  |  ".join([f"{b}: {p*100:.2f}%" for b, p in results])
                self.confidence_label.config(text=f"Top-{topk}: {conf_text}")

            # reset previous measurement display
            self.length_label.config(text="Body length: -")
            self.height_label.config(text="Height: -")
            self.angle_label.config(text="Rump angle: -")
            self.contour_label.config(text="Contour area (px): -")
            self.update_status()
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {e}")

    def load_model_dialog(self):
        model_file = filedialog.askopenfilename(title="Select model (.pth/.pt)", filetypes=[("PyTorch", "*.pth *.pt")])
        if not model_file:
            return
        labels_file = filedialog.askopenfilename(title="Select labels file (xlsx/csv)", filetypes=[("Excel/CSV", "*.xlsx *.csv")])
        if not labels_file:
            messagebox.showwarning("Labels missing", "No labels selected. Aborting model load.")
            return
        try:
            new_model, new_idx_to_breed = self.load_model_and_mapping(model_file, labels_file)
            self.model = new_model
            self.idx_to_breed = new_idx_to_breed
            self.model_loaded_from = model_file
            self.update_status()
            messagebox.showinfo("Success", f"Loaded model: {os.path.basename(model_file)}\nLabels: {os.path.basename(labels_file)}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model/labels: {e}")

    def batch_predict_folder(self):
        folder = filedialog.askdirectory(title="Select folder with images")
        if not folder:
            return
        out_csv = filedialog.asksaveasfilename(title="Save CSV as", defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not out_csv:
            return
        if self.model is None or not self.idx_to_breed:
            messagebox.showwarning("No model", "Load a model and labels first.")
            return

        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        files = sorted([f for f in files if os.path.isfile(f)])
        if not files:
            messagebox.showwarning("No images", "No images found in folder.")
            return

        with open(out_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "predicted_breed", "confidence"])
            for fp in files:
                try:
                    res = predict_breed_topk(fp, self.model, self.idx_to_breed, self.device, topk=1)
                    pred, conf = res[0]
                    writer.writerow([fp, pred, conf])
                except Exception as e:
                    writer.writerow([fp, "ERROR", str(e)])
        messagebox.showinfo("Batch done", f"Saved results to {out_csv}")

    # ------------------ Calibration --------------------
    def auto_calibrate_aruco(self):
        if not self.current_image_path:
            messagebox.showwarning("No image", "Select an image first.")
            return
        try:
            img_bgr = cv2.cvtColor(np.array(self.current_pil_image), cv2.COLOR_RGB2BGR)
            cm_per_px = self.measure.aruco_calibration_factor(img_bgr, marker_length_cm=10.0)  # default marker size 10 cm
            if cm_per_px is None:
                messagebox.showinfo("ArUco", "No ArUco marker detected in image.")
            else:
                self.calib_label.config(text=f"Calibration (cm/px): {cm_per_px:.6f}")
                self.update_status()
                messagebox.showinfo("ArUco", f"Calibration set: {cm_per_px:.6f} cm/px")
        except Exception as e:
            messagebox.showerror("ArUco error", str(e))

    def start_manual_calibration(self):
        if not self.current_pil_image:
            messagebox.showwarning("No image", "Select an image first.")
            return
        dialog = ManualCalibrationDialog(self.root, self.current_pil_image)
        cm_per_px = dialog.show()
        if cm_per_px:
            self.measure.calibration_factor = cm_per_px
            self.calib_label.config(text=f"Calibration (cm/px): {cm_per_px:.6f}")
            self.update_status()

    def on_image_click(self, event):
        # used for manual clicks if manual calibration mode toggled else ignore for now
        if not self.manual_cal_mode:
            return
        # Map click on displayed image back to original coordinates
        if not self.current_pil_image or not self.display_size:
            return
        dx, dy = self.display_size
        ox, oy = self.current_pil_image.size
        sx = ox / dx
        sy = oy / dy
        img_x = int(event.x * sx)
        img_y = int(event.y * sy)
        self.manual_clicks.append((img_x, img_y))
        # Draw small marker on thumbnail to give feedback
        # convert to display coords for drawing: event.x, event.y
        # draw a circle on the displayed image inside the label by creating an overlay image
        # (simple approach: place a temp dot image)
        # For now, just show count
        if len(self.manual_clicks) >= 2:
            (x1, y1), (x2, y2) = self.manual_clicks[:2]
            px_dist = np.hypot(x1 - x2, y1 - y2)
            val = simpledialog.askfloat("Enter real distance", "Enter actual distance between the two points (in cm):", parent=self.root)
            if val is None:
                messagebox.showinfo("Calibration", "Manual calibration cancelled.")
            else:
                cm_per_px = val / px_dist
                self.measure.calibration_factor = cm_per_px
                self.calib_label.config(text=f"Calibration (cm/px): {cm_per_px:.6f}")
                self.update_status()
            # reset manual calibration mode
            self.manual_clicks = []
            self.manual_cal_mode = False

    def extract_measurements_from_current(self):
        if not self.current_image_path or not self.current_pil_image:
            messagebox.showwarning("No image", "Select an image first.")
            return
        # convert PIL to BGR numpy
        img_bgr = cv2.cvtColor(np.array(self.current_pil_image), cv2.COLOR_RGB2BGR)
        # we do full-image extraction. If you have bbox from a detector, pass bbox here.
        measurements = self.measure.extract_measurements(img_bgr, bbox=None)
        if measurements is None:
            messagebox.showinfo("No contour", "Could not find animal contour or extract measurements.")
            return

        # display values (prefer cm if available)
        if measurements.get("body_length_cm") is not None:
            self.length_label.config(text=f"Body length: {measurements['body_length_cm']:.1f} cm")
        else:
            self.length_label.config(text=f"Body length: {measurements['body_length_px']:.1f} px")

        if measurements.get("height_cm") is not None:
            self.height_label.config(text=f"Height: {measurements['height_cm']:.1f} cm")
        else:
            self.height_label.config(text=f"Height: {measurements['height_px']:.1f} px")

        self.angle_label.config(text=f"Rump angle: {measurements.get('rump_angle', 'N/A')}")
        self.contour_label.config(text=f"Contour area (px): {measurements.get('contour_area_px', 0):.1f}")
        self.calib_label.config(text=f"Calibration (cm/px): {self.measure.calibration_factor:.6f}" if self.measure.calibration_factor else "Calibration (cm/px): -")
        # store last measurement for saving
        self.last_measurements = measurements
        messagebox.showinfo("Measurements", "Measurements extracted and shown in Results panel.")

    def save_current_evaluation(self):
        if not hasattr(self, "last_measurements") or not self.current_image_path:
            messagebox.showwarning("No results", "No measurements found. Run Extract Measurements first.")
            return
        detection = {}  # if you have detection output (bbox/class), fill here
        classification = {}
        # attempt to read predicted label from UI
        pred = self.result_label.cget("text").replace("Prediction:", "").strip()
        if pred and pred != "(no model loaded)":
            classification = {"breed": pred, "is_purebred": False}
        saved_path = self.data_manager.save_evaluation(
            image_path=self.current_image_path,
            detection=detection,
            measurements=self.last_measurements,
            classification=classification,
            metadata={"note": "Saved from GUI"}
        )
        messagebox.showinfo("Saved", f"Evaluation saved to:\n{saved_path}")

    def show_gradcam(self):
        if not getattr(self, "current_image_path", None):
            messagebox.showwarning("No image", "Select an image first.")
            return
        if self.model is None:
            messagebox.showwarning("No model", "Load a model first.")
            return
        try:
            overlay = self.generate_gradcam_local(self.current_image_path)
            top = tk.Toplevel(self.root)
            top.title("Grad-CAM")
            img_tk = ImageTk.PhotoImage(overlay)
            lbl = tk.Label(top, image=img_tk)
            lbl.image = img_tk
            lbl.pack()
        except Exception as e:
            messagebox.showerror("Grad-CAM error", str(e))

    def generate_gradcam_local(self, image_path, target_idx=None):
        """
        Local grad-cam hook similar to previous implementation.
        Hooks model.features[6].
        """
        model = self.model
        device = self.device
        img = Image.open(image_path).convert("RGB")
        x = transforms.Resize((IMG_SIZE, IMG_SIZE))(img)
        x_t = transforms.ToTensor()(x).unsqueeze(0).to(device)
        activations = []
        gradients = []

        def forward_hook(module, inp, out):
            activations.append(out.detach().cpu())
            def save_grad(grad):
                gradients.append(grad.detach().cpu())
            out.register_hook(save_grad)

        handle = model.features[6].register_forward_hook(forward_hook)
        out = model(x_t)
        probs = F.softmax(out[0], dim=0)
        if target_idx is None:
            target_idx = int(probs.argmax().item())
        model.zero_grad()
        one_hot = torch.zeros_like(out)
        one_hot[0, target_idx] = 1.0
        out.backward(gradient=one_hot)

        if len(activations) == 0 or len(gradients) == 0:
            handle.remove()
            raise RuntimeError("Grad-CAM failed to capture activations/gradients")

        act = activations[0][0].numpy()
        grad = gradients[0][0].numpy()
        weights = np.mean(grad, axis=(1, 2))
        cam = np.zeros(act.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * act[i]
        cam = np.maximum(cam, 0)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
        heatmap = (255 * cam).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        orig = np.array(img.resize((IMG_SIZE, IMG_SIZE)))
        overlay = (0.6 * orig + 0.4 * heatmap).astype(np.uint8)
        handle.remove()
        return Image.fromarray(overlay)

    def flag_incorrect(self):
        if not getattr(self, "current_image_path", None):
            messagebox.showwarning("No image", "Select an image first.")
            return
        predicted = self.result_label.cget("text").replace("Prediction: ", "").strip()
        correct = simpledialog.askstring("Correct label", "Enter correct breed label (or leave blank):", parent=self.root)
        row = {
            "image_path": self.current_image_path,
            "predicted": predicted,
            "correct": correct if correct else ""
        }
        header = not os.path.exists(RETRAIN_QUEUE)
        with open(RETRAIN_QUEUE, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "predicted", "correct"])
            if header:
                writer.writeheader()
            writer.writerow(row)
        messagebox.showinfo("Saved", f"Image added to {RETRAIN_QUEUE} for retraining.")

# ------------------ Main Execution -------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = BreedPredictorApp(root)
    root.mainloop()
