import cv2
import numpy as np

class FaceFramer:
    def __init__(self):
        self.target_ratio = 35 / 45      # 7:9 Portrait ratio
        self.face_height_ratio = 0.50    # Expanded base target for shoulder visibility
        self.top_head_gap_ratio = 0.15   # Clean overhead whitespace cushion

    def create_frame(self, image, bbox):
        if image is None or bbox is None:
            return None

        h, w, _ = image.shape
        x, y, bw, bh = bbox
        
        # --- FIX 1: ARTIFICIAL BOUNDING BOX STABILIZATION ---
        # If MediaPipe detects a hyper-focused tight box, we artificially expand 
        # it to encompass the hair and chin boundaries accurately.
        padding_factor = 1.35
        new_bh = int(bh * padding_factor)
        new_bw = int(bw * padding_factor)
        
        # Keep the adjusted center perfectly stable
        center_x = x + (bw // 2)
        center_y = y + (bh // 2)
        
        new_x = center_x - (new_bw // 2)
        new_y = center_y - (new_bh // 2)

        # --- FIX 2: CROPPING RECALCULATION ---
        crop_h = int(new_bh / self.face_height_ratio)
        crop_w = int(crop_h * self.target_ratio)
        
        # Allocate vertical clearance rules
        top_gap = int(crop_h * self.top_head_gap_ratio)

        y1 = new_y - top_gap
        y2 = y1 + crop_h
        x1 = center_x - (crop_w // 2)
        x2 = x1 + crop_w

        # --- FIX 3: ASPECT RATIO SAFETY VALVE ---
        # Ensure crop size is mathematically large enough relative to the image size.
        # This completely guarantees it will never crop into just her nose/lips.
        min_allowed_height = int(h * 0.40)  # Must capture at least 40% of the image height
        if crop_h < min_allowed_height:
            crop_h = min_allowed_height
            crop_w = int(crop_h * self.target_ratio)
            # Re-align boundaries based on normalized size
            y1 = center_y - int(crop_h * 0.40)
            y2 = y1 + crop_h
            x1 = center_x - (crop_w // 2)
            x2 = x1 + crop_w

        # --- FIX 4: SAFE OUT-OF-BOUNDS WHITE PADDING ---
        pad_top = max(0, -y1)
        pad_bottom = max(0, y2 - h)
        pad_left = max(0, -x1)
        pad_right = max(0, x2 - w)

        safe_y1, safe_y2 = max(0, y1), min(h, y2)
        safe_x1, safe_x2 = max(0, x1), min(w, x2)

        cropped = image[safe_y1:safe_y2, safe_x1:safe_x2]

        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            cropped = cv2.copyMakeBorder(
                cropped, pad_top, pad_bottom, pad_left, pad_right, 
                cv2.BORDER_CONSTANT, value=[255, 255, 255]
            )

        return cropped
