import cv2

class PassportResizer:
    def __init__(self):
        self.target_width=413
        self.target_height=531

    def resize(self,image):
        if image is None or image.size == 0:
            print("Error: Empty crop image passed to PassportResizer.")
            return None
        
        resized=cv2.resize(
            image,
            (self.target_width,self.target_height),
            interpolation=cv2.INTER_CUBIC
        )

        return resized