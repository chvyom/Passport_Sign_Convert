import cv2
import mediapipe as mp
import numpy as np

class PassportBackgroundChanger:
    def __init__(self, model_path: str = "selfie_segmenter.tflite"):
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            output_confidence_masks=True,
            output_category_mask=False
        )
        self.segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(options)

    def change_to_sky_blue(self, image):
        if image is None or image.size == 0:
            return None

        h, w, _ = image.shape
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Run segmentation
        segmentation_result = self.segmenter.segment(mp_image)
        
        # Grab the target confidence matrix
        raw_mask = segmentation_result.confidence_masks[0].numpy_view()

        # FIX: Drop any problematic extra dimensions
        clean_mask = np.squeeze(raw_mask)

        # FIX: Stack across channels to output an exact matching 3-dimensional shape (H, W, 3)
        alpha_3d = np.stack([clean_mask] * 3, axis=-1)

        # Initialize background canvas matching input photo size
        sky_blue_background = np.zeros(image.shape, dtype=np.uint8)
        sky_blue_background[:] = (235, 206, 135)

        # Convert to float to compute blending without math clipping errors
        foreground = image.astype(float)
        background = sky_blue_background.astype(float)

        # Blending operation works perfectly with equal matching shapes
        output_image = (foreground * alpha_3d) + (background * (1.0 - alpha_3d))
        
        return output_image.astype(np.uint8)
