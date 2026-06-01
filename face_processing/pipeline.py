import os
import cv2
from .detection import FaceDetector
from .bg_changer import PassportBackgroundChanger
from .framing import FaceFramer
from .resizer import PassportResizer


class FaceProcessingPipeline:

    def __init__(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Build absolute paths for BOTH model files
        face_model_path = os.path.join(current_dir, "blaze_face_full_range.tflite")
        segmenter_model_path = os.path.join(current_dir, "selfie_segmenter.tflite")
        
        # 3. Pass the paths into their respective constructors
        self.detector = FaceDetector(model_path=face_model_path)
        
        # Note: Ensure your PassportBackgroundChanger accepts a model_path parameter
        self.bg_changer = PassportBackgroundChanger(model_path=segmenter_model_path)
        
        self.framer = FaceFramer()
        self.resizer = PassportResizer()

    def is_already_valid_passport(self, image, bbox):
        
        img_h, img_w, _ = image.shape
        x, y, bw, bh = bbox

        current_ratio = img_w / img_h
        #target_ratio = 35 / 45
        face_height_percentage=bh/img_h

        #print(f"Image Dimensions: {img_w}x{img_h}")
        #print(f"Image Aspect Ratio: {current_ratio:.3f}")
        #print(f"Face Height Coverage: {face_height_percentage:.3f}")
        
        is_ratio_correct = 0.68 <= current_ratio <=0.88

        is_face_scaled_correctly = 0.35 <= face_height_percentage <= 0.58

        if is_ratio_correct and is_face_scaled_correctly:
            return True
            
        return False

    def process_image(self, image):
        if image is None or image.size == 0:
            return None
        
        original=image.copy()
        working_image=image.copy()

        initial_bbox = self.detector.detect_face(working_image)
        if initial_bbox is None:
            print("Pipeline skipped: No face found in the image.")
            return original

        if self.is_already_valid_passport(working_image, initial_bbox):
            return original.copy()

        print("Image format incorrect. Applying background color and cropping fixes...")

        blue_bg_image = self.bg_changer.change_to_sky_blue(working_image)
        if blue_bg_image is None:
            return None

        bbox = self.detector.detect_face(blue_bg_image)
        if bbox is None:
            return None

        cropped = self.framer.create_frame(blue_bg_image, bbox)

        final = self.resizer.resize(cropped)

        return final


    def process_batch(self, images):
        processed_output = []
        for image in images:
            result = self.process_image(image)
            if result is not None:
                processed_output.append(result)
        return processed_output


