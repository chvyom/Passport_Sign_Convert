import cv2
import numpy as np
import os
from typing import Tuple
import onnxruntime as ort

class PureSignatureValidator:
    def __init__(self, model_filename: str = "yolov8s.onnx"):
        """
        Hybrid AI + Structural Edge Bypass Engine.
        Guarantees validation even if the AI model fails to find a bounding box.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(current_dir, model_filename)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"CRITICAL: '{model_filename}' not found.")

        self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.input_width = 640
        self.input_height = 640

    def verify_and_process(self, image_path: str, margin: int = 15) -> Tuple[bool, str, np.ndarray]:
        if not os.path.exists(image_path):
            return False, "CRITICAL: Input file does not exist.", np.array([])

        try:
            with open(image_path, "rb") as f:
                img = cv2.imdecode(np.frombuffer(f.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None or img.size == 0:
                return False, "CRITICAL: Unreadable image structure.", np.array([])
        except Exception as e:
            return False, f"CRITICAL: Hardware read error: {str(e)}", np.array([])

        img_h, img_w = img.shape[:2]

        # --- STEP 1: RUN COMPUTER VISION FALLBACK IN PARALLEL ---
        # This isolates the exact coordinates of the ink strokes natively
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Threshold to catch dark ink on light background
        _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cv_x1, cv_y1, cv_x2, cv_y2 = img_w, img_h, 0, 0
        has_ink_structure = False
        
        for c in contours:
            cx, cy, cw, ch = cv2.boundingRect(c)
            # Ignore isolated noise specks smaller than 4x4 pixels
            if cw > 4 and ch > 4:
                has_ink_structure = True
                cv_x1 = min(cv_x1, cx)
                cv_y1 = min(cv_y1, cy)
                cv_x2 = max(cv_x2, cx + cw)
                cv_y2 = max(cv_y2, cy + ch)

        # --- STEP 2: PRE-PROCESS & RUN AI INFERENCE ---
        input_img = cv2.resize(img, (self.input_width, self.input_height))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = input_img.transpose(2, 0, 1)
        input_img = np.expand_dims(input_img, axis=0).astype(np.float32) / 255.0

        outputs = self.session.run(None, {self.input_name: input_img})
        predictions = np.squeeze(outputs)
        
        if len(predictions.shape) == 2 and predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.T

        conf_threshold = 0.25  
        boxes = []
        scores = []

        if len(predictions.shape) == 2 and predictions.shape[0] > 0:
            x_centers = predictions[:, 0]
            y_centers = predictions[:, 1]
            widths = predictions[:, 2]
            heights = predictions[:, 3]
            confidences = np.max(predictions[:, 4:], axis=1)

            raw_indices = np.where(confidences > conf_threshold)
            valid_indices = raw_indices[0] if len(raw_indices) > 0 else np.array([])

            scale_w, scale_h = img_w / 640.0, img_h / 640.0

            for idx in valid_indices:
                cx, cy, w, h = x_centers[idx], y_centers[idx], widths[idx], heights[idx]
                x1 = int(np.round((cx - w / 2.0) * scale_w))
                y1 = int(np.round((cy - h / 2.0) * scale_h))
                x2 = int(np.round((cx + w / 2.0) * scale_w))
                y2 = int(np.round((cy + h / 2.0) * scale_h))
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                scores.append(float(confidences[idx]))

        # --- STEP 3: DECISION MATRIX ---
        ai_detected = False
        w_box, h_box, confidence = 0, 0, 0.0

        if boxes:
            nms_indices = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=conf_threshold, nms_threshold=0.45)
            if len(nms_indices) > 0:
                best_idx = int(nms_indices.flatten()[0]) if isinstance(nms_indices, np.ndarray) else int(nms_indices[0])
                x_box, y_box, w_box, h_box = boxes[best_idx]
                confidence = scores[best_idx]
                ai_detected = True

        # Use AI data if available; otherwise drop back to native contour coordinates
        if ai_detected:
            is_already_correct = (w_box > img_w * 0.35) and (h_box > img_h * 0.15)
            if is_already_correct:
                return True, f"VERIFIED (AI): Image is already a clear close-up signature (Conf: {confidence:.2f}).", img
            x_min, y_min, x_max, y_max = x_box, y_box, x_box + w_box, y_box + h_box
        elif has_ink_structure:
            # Fallback path for when ONNX engine skips/misses clean white background images
            cv_w = cv_x2 - cv_x1
            cv_h = cv_y2 - cv_y1
            is_already_correct = (cv_w > img_w * 0.35) and (cv_h > img_h * 0.15)
            if is_already_correct:
                return True, "VERIFIED (CV-Fallback): Image structural analysis confirms a valid close-up signature.", img
            x_min, y_min, x_max, y_max = cv_x1, cv_y1, cv_x2, cv_y2
        else:
            return False, "VERIFICATION FAILED: No handwritten strokes or patterns found.", np.array([])

        # --- STEP 4: SAFE PADDED CROP & CLEANUP (For high-resolution canvas documents) ---
        crop_margin = max(30, margin)
        crop_x_min = max(0, x_min - crop_margin)
        crop_y_min = max(0, y_min - crop_margin)
        crop_x_max = min(img_w, x_max + crop_margin)
        crop_y_max = min(img_h, y_max + crop_margin)

        cropped_sig = img[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

        # White-balance background bleach transformation
        bg_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31))
        bg_map = cv2.dilate(cropped_sig, bg_kernel)
        processed_output = cv2.divide(cropped_sig, bg_map, scale=255)

        return True, "VERIFICATION SUCCESS: Signature isolated and background balanced.", processed_output
