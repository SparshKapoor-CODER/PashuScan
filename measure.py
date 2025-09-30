# measure.py
import cv2
import numpy as np
from scipy.spatial import distance
from datetime import datetime

class CattleMeasurements:
    """
    Robust measurement utilities:
      - GrabCut based contour extraction (fallback to Otsu)
      - Withers / tailhead estimation
      - Body length & height calculation (requires calibration factor: cm per pixel)
      - Rump angle estimation using ellipse fit
    """

    def __init__(self, calibration_factor=None):
        # calibration_factor = cm per pixel (cm/px). If None, measurement returns pixel units until set.
        self.calibration_factor = calibration_factor

    def set_calibration(self, pixels, actual_cm):
        """Set calibration factor from measured px distance and actual distance in cm"""
        if pixels <= 0:
            raise ValueError("Pixel distance must be > 0")
        self.calibration_factor = actual_cm / pixels
        return self.calibration_factor

    def pixels_to_cm(self, px):
        if self.calibration_factor is None:
            return None
        return float(px * self.calibration_factor)

    # ---------- ArUco calibration helper ----------
    def aruco_calibration_factor(self, image_bgr, marker_length_cm=10.0, aruco_dict_type=cv2.aruco.DICT_4X4_50):
        """
        Detects an ArUco marker in the image and returns cm-per-pixel factor (cm/px).
        Returns None if not found.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        try:
            # Newer OpenCV (4.7+) has Detector class
            aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
            parameters = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, _ = detector.detectMarkers(gray)
        except Exception:
            # Older style
            aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
            parameters = cv2.aruco.DetectorParameters_create()
            corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if corners is None or len(corners) == 0:
            return None

        # Use first marker (average side length in px)
        c = corners[0].reshape((4, 2))
        side_px = np.mean([
            np.linalg.norm(c[0] - c[1]),
            np.linalg.norm(c[1] - c[2]),
            np.linalg.norm(c[2] - c[3]),
            np.linalg.norm(c[3] - c[0])
        ])
        # px per cm
        px_per_cm = side_px / float(marker_length_cm)
        # we store calibration as cm/px (consistent with set_calibration)
        cm_per_px = 1.0 / px_per_cm
        self.calibration_factor = cm_per_px
        return self.calibration_factor

    # ---------- Contour extraction ----------
    def extract_contour_from_bbox(self, image_bgr, bbox):
        """
        bbox: (x1,y1,x2,y2) ints in image coordinates
        returns (contour, (x_offset, y_offset)) or (None, None)
        """
        x1, y1, x2, y2 = bbox
        h, w = image_bgr.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        roi = image_bgr[y1:y2, x1:x2].copy()

        if roi.size == 0:
            return None, (x1, y1)

        # GrabCut initialization rectangle (avoid edges)
        rect = (5, 5, max(1, roi.shape[1] - 10), max(1, roi.shape[0] - 10))
        mask = np.zeros(roi.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        try:
            cv2.grabCut(roi, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            grab_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8') * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            grab_mask = cv2.morphologyEx(grab_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
            grab_mask = cv2.morphologyEx(grab_mask, cv2.MORPH_OPEN, kernel, iterations=1)
            contours, _ = cv2.findContours(grab_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None, (x1, y1)
            largest = max(contours, key=cv2.contourArea)
            return largest, (x1, y1)
        except Exception:
            # fallback to Otsu threshold
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None, (x1, y1)
            return max(contours, key=cv2.contourArea), (x1, y1)

    def extract_contour_full_image(self, image_bgr):
        """
        Attempt to extract the main animal contour using GrabCut over the entire image.
        Returns (contour, (0,0)) or (None, None)
        """
        h, w = image_bgr.shape[:2]
        rect = (5, 5, max(1, w - 10), max(1, h - 10))
        mask = np.zeros((h, w), np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(image_bgr, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8') * 255
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask2 = cv2.morphologyEx(mask2, cv2.MORPH_CLOSE, kernel, iterations=2)
            contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None, (0, 0)
            largest = max(contours, key=cv2.contourArea)
            return largest, (0, 0)
        except Exception:
            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return None, (0, 0)
            return max(contours, key=cv2.contourArea), (0, 0)

    # ---------- Estimators ----------
    def estimate_withers(self, contour, offset=(0, 0)):
        x_off, y_off = offset
        topmost = tuple(contour[contour[:, :, 1].argmin()][0])
        return (int(topmost[0] + x_off), int(topmost[1] + y_off))

    def estimate_tailhead(self, contour, offset=(0, 0)):
        x_off, y_off = offset
        bottommost = tuple(contour[contour[:, :, 1].argmax()][0])
        return (int(bottommost[0] + x_off), int(bottommost[1] + y_off))

    def calculate_body_length_px(self, withers, tailhead):
        return float(distance.euclidean(withers, tailhead))

    def calculate_height_px(self, withers, bbox_bottom_y):
        return abs(float(withers[1] - bbox_bottom_y))

    def analyze_rump_angle(self, contour, offset=(0, 0)):
        # fit ellipse if enough points
        try:
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                angle = ellipse[2]
                # normalize angle to [0,90]
                if angle > 90:
                    angle = 180 - angle
                return float(angle)
            else:
                return None
        except Exception:
            return None
        
    def estimate_chest_width(self, contour, offset=(0, 0)):
        """
        Estimate chest width from contour by finding widest point
        """
        x_off, y_off = offset
        
        try:
            # Find the bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Estimate chest area (middle third of body height)
            chest_y_start = y + h // 3
            chest_y_end = y + 2 * h // 3
            
            # Find leftmost and rightmost points in chest region
            chest_points = []
            for point in contour:
                px, py = point[0]
                if chest_y_start <= (py + y_off) <= chest_y_end:
                    chest_points.append((px + x_off, py + y_off))
            
            if len(chest_points) < 2:
                return {"chest_width_px": 0.0, "chest_width_cm": None}
            
            # Find maximum width in chest region
            chest_points = np.array(chest_points)
            min_x = np.min(chest_points[:, 0])
            max_x = np.max(chest_points[:, 0])
            
            chest_width_px = float(max_x - min_x)
            chest_width_cm = self.pixels_to_cm(chest_width_px)
            
            return {
                "chest_width_px": chest_width_px,
                "chest_width_cm": chest_width_cm
            }
        except Exception as e:
            print(f"Chest width measurement error: {e}")
            return {"chest_width_px": 0.0, "chest_width_cm": None}
        

    # ---------- Main extraction ----------
    def extract_measurements(self, image_bgr, bbox=None):
        """
        image_bgr: full color image (numpy BGR)
        bbox: optional bounding box in image coords [x1,y1,x2,y2]
        Returns a dict with measurements. Values in cm if calibration_factor set, else px.
        """
        # Choose contour extraction method
        if bbox is not None:
            contour, offset = self.extract_contour_from_bbox(image_bgr, bbox)
            x1, y1, x2, y2 = bbox
            bbox_bottom = y2
        else:
            contour, offset = self.extract_contour_full_image(image_bgr)
            bbox_bottom = image_bgr.shape[0] - 1

        if contour is None:
            return None

        withers = self.estimate_withers(contour, offset)
        tailhead = self.estimate_tailhead(contour, offset)
        body_px = self.calculate_body_length_px(withers, tailhead)
        height_px = self.calculate_height_px(withers, bbox_bottom)
        rump_angle = self.analyze_rump_angle(contour, offset)
        contour_area = float(cv2.contourArea(contour))

        body_cm = self.pixels_to_cm(body_px) if self.calibration_factor else None
        height_cm = self.pixels_to_cm(height_px) if self.calibration_factor else None

        return {
            "withers_position": (int(withers[0]), int(withers[1])),
            "tailhead_position": (int(tailhead[0]), int(tailhead[1])),
            "body_length_px": float(body_px),
            "height_px": float(height_px),
            "body_length_cm": body_cm,
            "height_cm": height_cm,
            "rump_angle": rump_angle,
            "contour_area_px": contour_area,
            "calibration_factor_cm_per_px": self.calibration_factor
        }
