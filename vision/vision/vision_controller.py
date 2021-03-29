import numpy as np

class ObjectInfo:
    def __init__(self):
        self.mask_full = None
        self.mask_cropped = None
        self.object_img_cutout_full = None
        self.object_img_cutout_cropped = None
        self.bbox_xxyy = None
        self.bbox_xywh = None
        self.features = None

class VisionController():
    def get_masks_with_features(self):  # dummy function so i can mock it
        features = []
        object_info = ObjectInfo()
        features.append(object_info)
        return features

if __name__ == "__main__":
    vision = VisionController()
    print(vision.get_masks_with_features())