import cv2
import numpy as np
import os
from skimage.measure import label, regionprops

# ─────────────────────────────────────────────────────────────────────────────
# YOLOv8 ONNX — PRIMARY DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

def load_yolo_model(model_path: str):
    """Load the YOLOv8 ONNX model with ONNXRuntime. Returns session or None."""
    try:
        import onnxruntime as ort
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        session = ort.InferenceSession(model_path, providers=providers)
        print(f"✅ YOLO model loaded: {model_path}")
        return session
    except Exception as e:
        print(f"⚠️  Could not load ONNX model ({e}). Will use statistical fallback.")
        return None


def preprocess_for_yolo(img_bgr: np.ndarray, input_size: int = 640):
    """
    Letterbox-resize to input_size × input_size, normalize to [0,1],
    convert HWC-BGR → CHW-RGB, add batch dim.
    Returns (blob, scale, pad_x, pad_y).
    """
    h, w = img_bgr.shape[:2]
    scale = min(input_size / h, input_size / w)
    nh, nw = int(round(h * scale)), int(round(w * scale))

    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_x = (input_size - nw) // 2
    pad_y = (input_size - nh) // 2

    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    canvas[pad_y:pad_y + nh, pad_x:pad_x + nw] = resized

    blob = canvas[:, :, ::-1].astype(np.float32) / 255.0   # BGR→RGB, normalise
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis]        # HWC → 1CHW
    return blob, scale, pad_x, pad_y


def postprocess_yolo(output: np.ndarray, scale: float, pad_x: int, pad_y: int,
                     orig_h: int, orig_w: int,
                     conf_threshold: float = 0.25, iou_threshold: float = 0.45):
    """
    YOLOv8 raw output shape: [1, 5, num_anchors]  (x_c, y_c, w, h, conf).
    Returns list of (x1, y1, x2, y2, confidence) in original-image pixels.
    """
    preds = output[0]           # → [5, num_anchors]
    preds = preds.T             # → [num_anchors, 5]

    boxes_xywh = preds[:, :4]
    confs      = preds[:, 4]

    keep = confs >= conf_threshold
    boxes_xywh = boxes_xywh[keep]
    confs       = confs[keep]

    if len(confs) == 0:
        return []

    # Convert cx, cy, w, h (letterbox space) → x1y1x2y2 (original image space)
    cx, cy, bw, bh = boxes_xywh[:, 0], boxes_xywh[:, 1], boxes_xywh[:, 2], boxes_xywh[:, 3]
    x1 = (cx - bw / 2 - pad_x) / scale
    y1 = (cy - bh / 2 - pad_y) / scale
    x2 = (cx + bw / 2 - pad_x) / scale
    y2 = (cy + bh / 2 - pad_y) / scale

    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)

    # NMS
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xyxy.tolist(),
        scores=confs.tolist(),
        score_threshold=conf_threshold,
        nms_threshold=iou_threshold,
    )

    results = []
    if len(indices) > 0:
        for i in np.array(indices).flatten():
            results.append((
                int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i]), float(confs[i])
            ))
        results.sort(key=lambda r: r[4], reverse=True)   # highest confidence first

    return results


def detect_with_yolo(session, image_path: str, conf_threshold: float = 0.25):
    """
    Run YOLOv8 inference on one image.
    Returns (x, y, crop_w, crop_h) of the best detection, or None.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    blob, scale, pad_x, pad_y = preprocess_for_yolo(img)

    input_name = session.get_inputs()[0].name
    raw_output = session.run(None, {input_name: blob})

    detections = postprocess_yolo(
        raw_output[0], scale, pad_x, pad_y, h, w, conf_threshold
    )

    if not detections:
        return None

    x1, y1, x2, y2, conf = detections[0]   # best detection
    print(f"   🎯 YOLO detected signature  conf={conf:.2f}  box=({x1},{y1},{x2},{y2})")
    return (x1, y1, x2 - x1, y2 - y1)


# ─────────────────────────────────────────────────────────────────────────────
# STATISTICAL FALLBACK — original sign.py logic
# ─────────────────────────────────────────────────────────────────────────────

def run_signature_detect_logic(image_path, outlier_weight=3.0, outlier_bias=100,
                                amplifier=15, min_area_size=10):
    """
    HSV-mask + skimage region-labeling fallback.
    Returns (x, y, crop_w, crop_h) or None.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    low_threshold  = np.array([0,   0,   0])
    high_threshold = np.array([180, 255, 200])
    mask = cv2.inRange(hsv, low_threshold, high_threshold)

    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)

    if not regions:
        return None

    total_pixels, nb_regions, valid_regions = 0, 0, []
    for prop in regions:
        if prop.area >= min_area_size:
            total_pixels += prop.area
            nb_regions   += 1
            valid_regions.append(prop)

    if nb_regions == 0:
        return None

    average_size       = total_pixels / nb_regions
    small_size_outlier = average_size * outlier_weight + outlier_bias
    big_size_outlier   = small_size_outlier * amplifier

    x_min, y_min = w, h
    x_max, y_max = 0, 0
    strokes_found = False

    for prop in valid_regions:
        if small_size_outlier <= prop.area <= big_size_outlier:
            minr, minc, maxr, maxc = prop.bbox
            x_min = min(x_min, minc)
            y_min = min(y_min, minr)
            x_max = max(x_max, maxc)
            y_max = max(y_max, maxr)
            strokes_found = True

    if strokes_found:
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CORE CROP + ENHANCE HELPER  (shared by both the class and standalone use)
# ─────────────────────────────────────────────────────────────────────────────

def _crop_and_enhance(img: np.ndarray, box) -> np.ndarray:
    """
    Given a source BGR image and a (x, y, cw, ch) box (or None for full-frame),
    apply margin padding, 1.2× upscale, and bilateral smoothing.
    Returns the final enhanced BGR array.
    """
    h, w = img.shape[:2]

    if box is None:
        x, y, cw, ch = 0, 0, w, h
    else:
        x, y, cw, ch = box

    margin_w = int(cw * 0.05) if w > 500 else 8
    margin_h = int(ch * 0.05) if h > 500 else 8

    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(w, x + cw + margin_w)
    y2 = min(h, y + ch + margin_h)

    cropped = img[y1:y2, x1:x2]
    if cropped.size == 0:
        cropped = img

    scale  = 1.2
    new_w  = int(cropped.shape[1] * scale)
    new_h  = int(cropped.shape[0] * scale)
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return cv2.bilateralFilter(resized, 5, 25, 25)


# ─────────────────────────────────────────────────────────────────────────────
# PureSignatureValidator — interface consumed by main.py
# ─────────────────────────────────────────────────────────────────────────────

class PureSignatureValidator:
    """
    Drop-in class for main.py.

    Usage:
        validator = PureSignatureValidator(model_path="yolov8s.onnx")
        success, log_msg, result_img = validator.verify_and_process(file_path)

    Returns:
        success   (bool)        – True if a signature region was confidently found
        log_msg   (str)         – Human-readable status string for logging/UI
        result_img (np.ndarray) – Enhanced crop (or full frame as fallback); never None
                                  unless the file itself could not be read.
    """

    def __init__(self, model_path: str = None, conf_threshold: float = 0.25):
        # Resolve model path: explicit arg → env var → sibling file
        if model_path is None:
            model_path = os.environ.get(
                "MODEL_PATH",
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolov8s.onnx")
            )
        self.conf_threshold = conf_threshold
        self.session = load_yolo_model(model_path)   # None if unavailable

    # ------------------------------------------------------------------
    def verify_and_process(self, image_path: str):
        """
        Detect, crop, and enhance the signature in *image_path*.

        Returns (success: bool, log_msg: str, result_img: np.ndarray | None).
        """
        img = cv2.imread(image_path)
        if img is None:
            return False, f"❌ Cannot read image: {image_path}", None

        h, w = img.shape[:2]
        box         = None
        success     = False
        method_used = "whole-frame"

        # ── 1. YOLOv8 ────────────────────────────────────────────────────────
        if self.session is not None:
            box = detect_with_yolo(self.session, image_path, self.conf_threshold)
            if box is not None:
                success     = True
                method_used = "YOLO"

        # ── 2. Statistical fallback ──────────────────────────────────────────
        if box is None:
            box = run_signature_detect_logic(image_path)
            if box is not None:
                success     = True
                method_used = "statistical"

        # ── 3. Whole-frame last resort ───────────────────────────────────────
        if box is None:
            box         = (0, 0, w, h)
            success     = False       # no real detection; flag caller
            method_used = "whole-frame"

        result_img = _crop_and_enhance(img, box)
        log_msg    = f"[{method_used}] {'detected' if success else 'no detection — full frame used'}"
        return success, log_msg, result_img


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE ENTRY POINT  (python sign.py)
# ─────────────────────────────────────────────────────────────────────────────

def process_signature(image_path: str, session, output_folder: str = "output",
                      conf_threshold: float = 0.25):
    """Standalone helper that saves the result to disk (used by __main__)."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Cannot load {image_path}")
        return False

    h, w   = img.shape[:2]
    box    = None
    method = "whole-frame"

    if session is not None:
        box = detect_with_yolo(session, image_path, conf_threshold)
        if box:
            method = "YOLO"

    if box is None:
        print(f"   ↩️  YOLO found nothing — trying statistical fallback …")
        box = run_signature_detect_logic(image_path)
        if box:
            method = "statistical"

    if box is None:
        print(f"   ⚠️  No signature region found. Using whole frame.")
        box = (0, 0, w, h)

    final_output = _crop_and_enhance(img, box)

    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, f"DETECT_{os.path.basename(image_path)}")
    cv2.imwrite(output_path, final_output)
    print(f"   ✅ [{method}] saved → {output_path}")
    return True


if __name__ == "__main__":
    current_folder = os.path.dirname(os.path.abspath(__file__))
    model_path     = os.environ.get("MODEL_PATH",
                                    os.path.join(current_folder, "yolov8s.onnx"))
    test_folder    = os.path.join(current_folder, "test_images")
    os.makedirs(test_folder, exist_ok=True)

    files = [f for f in os.listdir(test_folder)
             if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]

    if not files:
        print(f"📁 Put your images inside: {test_folder}")
    else:
        session = load_yolo_model(model_path)
        print("=" * 60)
        print("🔍 SIGNATURE DETECTION  (YOLOv8 → statistical fallback)")
        print("=" * 60)
        for file in files:
            print(f"\n📷 {file}")
            process_signature(os.path.join(test_folder, file), session)