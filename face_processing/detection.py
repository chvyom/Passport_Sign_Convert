import cv2
import mediapipe as mp

class FaceDetector:
    def __init__(self,model_path: str="blaze_face_full_range.tflite"):
        base_options=mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.FaceDetectorOptions(
            base_options=base_options,
            min_detection_confidence=0.6
        )
        # Initialize the Task detector
        self.detector = mp.tasks.vision.FaceDetector.create_from_options(options)

    def detect_face(self,image):
        h,w,_=image.shape

        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB, 
            data=rgb
        )

        results=self.detector.detect(mp_image)

        if not results.detections:
            return None
        
        detection=max(
            results.detections,
            key=lambda d: d.bounding_box.width * d.bounding_box.height
        )

        bbox=detection.bounding_box

        x = max(0, int(bbox.origin_x))
        y = max(0, int(bbox.origin_y))
        bw = min(int(bbox.width), w - x)
        bh = min(int(bbox.height), h - y)

        return (x,y,bw,bh)

